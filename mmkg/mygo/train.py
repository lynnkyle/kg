import os
import sys
import argparse

import numpy as np
from tqdm import tqdm

import torch
import logging

from dataset import VTKG
from model_mygo import MyGo
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens
from utils import calculate_rank, metrics

"""
    参数设置
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='MKG-W')
parser.add_argument('--model', type=str, default='MyGo')
parser.add_argument('--device', type=int, default='cuda')
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--valid_epoch', type=int, default=50)
parser.add_argument('--str_dim', default=256, type=int)
parser.add_argument("--no_write", action='store_true')
parser.add_argument('--str_dropout', default=0.6, type=float)
parser.add_argument('--visual_dropout', default=0.3, type=float)
parser.add_argument('--textual_dropout', default=0.1, type=float)
parser.add_argument('--fgcl_weight', default=0.01, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
# Transformer的配置
parser.add_argument('--num_head', default=2, type=int)
parser.add_argument('--dim_hid', default=1024, type=int)
parser.add_argument('--num_layer_enc_ent', default=1, type=int)
parser.add_argument('--num_layer_enc_rel', default=1, type=int)
parser.add_argument('--num_layer_dec', default=1, type=int)
parser.add_argument('--dropout', default=0.01, type=float)
args = parser.parse_args()

"""
    文件保存
"""
if not args.no_write:
    os.makedirs(f'result/{args.model}/{args.data}', exist_ok=True)
    os.makedirs(f'ckpt/{args.model}/{args.data}', exist_ok=True)
    os.makedirs(f'log/{args.model}/{args.data}', exist_ok=True)

"""
    日志输出
"""
logger = logging.getLogger('mygo')
logger.setLevel(logging.INFO)
format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(format)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler(f'log/{args.model}/{args.data}')
file_handler.setFormatter(format)
logger.addHandler(file_handler)

"""
    创建数据集
"""
kg = VTKG(data='MKG-W', max_vis_len=-1)
kg_loader = torch.utils.data.DataLoader(kg, batch_size=32, shuffle=True)

"""
    模型要素
"""
visual_token_index, visual_ent_mask = get_entity_visual_tokens(args.data, max_num=8)
textual_token_index, textual_ent_mask = get_entity_textual_tokens(args.data, max_num=4)
model = MyGo(num_ent=kg.num_ent, num_rel=kg.num_rel, str_dim=args.str_dim, visual_tokenizer='beit',
             textual_tokenizer='bert', visual_token_index=visual_token_index, textual_token_index=textual_token_index,
             visual_ent_mask=visual_ent_mask, textual_ent_mask=textual_ent_mask, num_head=args.num_head,
             dim_hid=args.dim_hid, num_layer_enc_ent=args.num_layer_enc_ent, num_layer_enc_rel=args.num_layer_enc_rel,
             num_layer_dec=args.num_layer_dec, dropout=args.dropout, str_dropout=args.str_dropout,
             visual_dropout=args.visual_dropout, textual_dropout=args.textual_dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

"""
    模型训练
"""


def train_one_epoch(model, optimizer):
    model.train()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch, label in tqdm(kg_loader):
        ent_embs, rel_embs = model()
        score = model.score(batch, ent_embs, rel_embs)
        loss = loss_fn(score, label)
        if args.fgcl_weight != 0:
            loss += args.mu * model.contrastive_loss_finegrained(ent_embs)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
    return total_loss


@torch.no_grad()
def valid_eval_metric(valid_or_test):
    model.eval()
    rank_list = []
    for triple in tqdm(valid_or_test):
        h, r, t = triple
        ent_embs, rel_embs = model()
        head_score = model.score(torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]), ent_embs,
                                 rel_embs)  # [batch_size, num_entity]
        head_rank = calculate_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        rank_list.append(head_rank)
        tail_score = model.score(torch.tensor([[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]), ent_embs,
                                 rel_embs)  # [batch_size, num_entity]
        tail_rank = calculate_rank(tail_score, r, kg.filter_dict[(h, r, -1)])
        rank_list.append(tail_rank)
    rank_list = np.array(rank_list)
    mr, mrr, hit10, hit3, hit1 = metrics(rank_list)
    return mr, mrr, hit10, hit3, hit1


best_mrr = 0
for epoch in range(args.num_epoch):
    loss = train_one_epoch(model, optimizer)
    lr_scheduler.step()
    logger.info(f'Epoch {epoch + 1}/{args.num_epoch}, Loss: {loss:.4f}')
    if (epoch + 1) % args.valid_epoch == 0:
        mr, mrr, hit10, hit3, hit1 = valid_eval_metric(valid_or_test=kg.valid)
        logger.info("Entity Prediction on Valid Set")
        logger.info(f"MR: {mr}")
        logger.info(f"MRR: {mrr}")
        logger.info(f"Hit10: {hit10}")
        logger.info(f"Hit3: {hit3}")
        logger.info(f"Hit1: {hit1}")
        mr, mrr, hit10, hit3, hit1 = valid_eval_metric(valid_or_test=kg.test)
        logger.info("Entity Prediction on Test Set")
        logger.info(f"MR: {mr}")
        logger.info(f"MRR: {mrr}")
        logger.info(f"Hit10: {hit10}")
        logger.info(f"Hit3: {hit3}")
        logger.info(f"Hit1: {hit1}")
        if mrr > best_mrr:
            best_mrr = mrr
            best_result = (mr, mrr, hit10, hit3, hit1)
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict()}, f'ckpt/{args.model}/{args.data}/{epoch}.ckpt')

logger.info(f'Best MRR: {best_mrr}')
logger.info("Done")
