import os
import random
import argparse
import numpy as np
import logging
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from dataset import VTKG
from torch.utils.data import DataLoader
from vista import VISTA
from utils import calculate_rank, metrics

"""
    代码可复现
"""
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(8)
torch.cuda.manual_seed_all(0)
torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
    参数设置
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='FB15K237', type=str)
parser.add_argument('--model', default='VISTA', type=str)
parser.add_argument('--ent_max_vis_len', default=3, type=int)
parser.add_argument('--rel_max_vis_len', default=3, type=int)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument('--valid_epoch', default=50, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dim_str', default=256, type=int)
parser.add_argument('--str_dropout', default=0.9, type=int)
parser.add_argument('--vis_dropout', default=0.4, type=int)
parser.add_argument('--txt_dropout', default=0.1, type=int)
parser.add_argument('--smoothing', default=0.0, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--step_size', default=50, type=int)
parser.add_argument('--no_write', action='store_true')
# Transformer网络参数
parser.add_argument('--num_head', default=4, type=int)
parser.add_argument('--dim_ffn', default=2048, type=int)
parser.add_argument('--num_layer_enc_ent', default=2, type=int)
parser.add_argument('--num_layer_enc_rel', default=1, type=int)
parser.add_argument('--num_layer_dec', default=2, type=int)
parser.add_argument('--dropout', default=0.01, type=int)
args = parser.parse_args()

"""
    文件保存
"""
if not args.no_write:
    os.makedirs(f"./result/{args.model}/{args.data}", exist_ok=True)
    os.makedirs(f"./ckpt/{args.model}/{args.data}", exist_ok=True)
    os.makedirs(f"./logs/{args.model}/{args.data}", exist_ok=True)

"""
    日志输出
"""
logger = logging.getLogger('vista')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
file_handler = logging.FileHandler(f"./logs/{args.model}/{args.data}/log.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

"""
    模型要素
"""
# 数据集
kg = VTKG(data=args.data, ent_max_vis_len=args.ent_max_vis_len, rel_max_vis_len=args.rel_max_vis_len)
kg_loader = DataLoader(kg, batch_size=args.batch_size, shuffle=True)
# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 模型
model = VISTA(kg.num_ent, kg.num_rel, args.dim_str, ent_vis=kg.ent_vis_matrix, rel_vis=kg.rel_vis_matrix,
              dim_vis=kg.vis_feat_dim, ent_txt=kg.ent_txt_matrix, rel_txt=kg.rel_txt_matrix, dim_txt=kg.txt_feat_dim,
              ent_vis_mask=kg.ent_vis_mask, rel_vis_mask=kg.rel_vis_mask, num_head=args.num_head, dim_ffn=args.dim_ffn,
              num_layer_enc_ent=args.num_layer_enc_ent, num_layer_enc_rel=args.num_layer_enc_rel,
              num_layer_dec=args.num_layer_dec, dropout=args.dropout, str_dropout=args.str_dropout,
              vis_dropout=args.vis_dropout, txt_dropout=args.txt_dropout).to(device)
state_dict = torch.load('/home/ps/lzy/kg/mmkg/vista/ckpt/VISTA/FB15K237/130.ckpt')
model.load_state_dict(state_dict['model_state_dict'])
# 损失函数
loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
# 优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer.load_state_dict(state_dict['optimizer_state_dict'])
# 学习率调度器
"   一般来说，学习率是按 epoch 进行调整的，而不是按每个 batch 调整  "
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.step_size, T_mult=2)
scheduler.load_state_dict(state_dict['scheduler_state_dict'])

"""
    模型训练
"""
logger.info("Start training...")
best_mrr = 0
last_epoch = 0
# model.train()
for epoch in range(last_epoch + 1, args.num_epoch + 1):
    total_loss = 0
    for batch, label in tqdm(kg_loader):
        ent_embs, rel_embs = model()
        scores = model.score(ent_embs, rel_embs, batch.cuda())
        loss = loss_fn(scores, label.cuda())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    scheduler.step()
    logger.info(f"Epoch {epoch}, loss: {total_loss}")
    if epoch % args.valid_epoch == 0:
        model.eval()
        with torch.no_grad():
            ent_embs, rel_embs = model()
            list_rank = []
            for triple in tqdm(kg.valid):
                h, r, t = triple
                head_score = model.score(ent_embs, rel_embs,
                                         torch.tensor(
                                             [[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda())[
                    0].detach().cpu().numpy()
                head_rank = calculate_rank(head_score, h, kg.filter_dict[(-1, r, t)])
                tail_score = model.score(ent_embs, rel_embs,
                                         torch.tensor(
                                             [[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda())[
                    0].detach().cpu().numpy()
                tail_rank = calculate_rank(tail_score, t, kg.filter_dict[(h, r, -1)])
                list_rank.append(head_rank)
                list_rank.append(tail_rank)
            list_rank = np.array(list_rank)
            mr, mrr, hit10, hit3, hit1 = metrics(list_rank)
            logger.info("Entity Prediction on Validation Set")
            logger.info(f"MR: {mr:.4f}")
            logger.info(f"MRR: {mrr:.4f}")
            logger.info(f"Hit@10: {hit10:.4f}")
            logger.info(f"Hit@3: {hit3:.4f}")
            logger.info(f"Hit@1: {hit1:.4f}")
        if best_mrr < mrr:
            best_mrr = mrr
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()},
                       f'./ckpt/{args.model}/{args.data}/{130 + epoch}.ckpt')
        model.train()
logger.info("Done")
