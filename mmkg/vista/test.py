import argparse
import random
import numpy as np
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataset import VTKG
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
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--best_epoch', default=30, type=int)
parser.add_argument('--dim_str', default=256, type=int)
parser.add_argument('--str_dropout', default=0.9, type=int)
parser.add_argument('--vis_dropout', default=0.4, type=int)
parser.add_argument('--txt_dropout', default=0.1, type=int)
parser.add_argument('--weight_decay', default=0.0, type=float)
# Transformer网络参数
parser.add_argument('--num_head', default=4, type=int)
parser.add_argument('--dim_ffn', default=2048, type=int)
parser.add_argument('--num_layer_enc_ent', default=2, type=int)
parser.add_argument('--num_layer_enc_rel', default=1, type=int)
parser.add_argument('--num_layer_dec', default=2, type=int)
parser.add_argument('--dropout', default=0.01, type=int)
args = parser.parse_args()

"""
    日志输出
"""
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
steam_handler = logging.StreamHandler()
steam_handler.setFormatter(log_format)
logger.addHandler(steam_handler)

"""
    模型要素
"""
# 数据集
kg = VTKG(args.data, ent_max_vis_len=args.ent_max_vis_len, rel_max_vis_len=args.rel_max_vis_len)
kg_loader = DataLoader(dataset=kg, batch_size=args.batch_size, shuffle=True)
# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 模型
model = VISTA(kg.num_ent, kg.num_rel, args.dim_str, ent_vis=kg.ent_vis_matrix, rel_vis=kg.rel_vis_matrix,
              dim_vis=kg.vis_feat_dim, ent_txt=kg.ent_txt_matrix, rel_txt=kg.rel_txt_matrix, dim_txt=kg.txt_feat_dim,
              ent_vis_mask=kg.ent_vis_mask, rel_vis_mask=kg.rel_vis_mask, num_head=args.num_head, dim_ffn=args.dim_ffn,
              num_layer_enc_ent=args.num_layer_enc_ent, num_layer_enc_rel=args.num_layer_enc_rel,
              num_layer_dec=args.num_layer_dec, dropout=args.dropout, str_dropout=args.str_dropout,
              vis_dropout=args.vis_dropout, txt_dropout=args.txt_dropout).to(device)

loaded_ckpt = torch.load(f'./ckpt/{args.model}/{args.data}/{args.best_epoch}.ckpt')
model.load_state_dict(loaded_ckpt['model_state_dict'])

"""
    模型测试
"""
model.eval()
with torch.no_grad():
    ent_embs, rel_embs = model()
    rank_list = []
    for triples in tqdm(kg.test):
        h, r, t = triples
        head_score = model.score(ent_embs, rel_embs,
                                 torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda())[
            0].detach().cpu().numpy()
        head_rank = calculate_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        tail_score = model.score(ent_embs, rel_embs,
                                 torch.tensor([[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda())[
            0].detach().cpu().numpy()
        tail_rank = calculate_rank(tail_score, t, kg.filter_dict[(h, r, -1)])
        rank_list.append(head_rank)
        rank_list.append(tail_rank)
    rank_list = np.array(rank_list)
    mr, mrr, hit10, hit3, hit1 = metrics(rank_list)
    logger.info(f'MR: {mr:.4f}')
    logger.info(f'MRR: {mrr:.4f}')
    logger.info(f'Hit@10: {hit10:.4f}')
    logger.info(f'Hit@3: {hit3:.4f}')
    logger.info(f'Hit@1: {hit1:.4f}')
