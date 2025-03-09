import os
import sys
import argparse

import torch
import logging

from dataset import VTKG
from model_mygo import MyGo
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens

"""
    参数设置
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='MKG-W')
parser.add_argument('--model', type=str, default='MyGo')
parser.add_argument('--str_dim', default=256, type=int)
parser.add_argument("--no_write", action='store_true')
parser.add_argument('--str_dropout', default=0.6, type=float)
parser.add_argument('--visual_dropout', default=0.3, type=float)
parser.add_argument('--textual_dropout', default=0.1, type=float)
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
# if not args.no_write:
#     os.makedirs(f'result/{args.model}/{args.data}', exist_ok=True)
#     os.makedirs(f'ckpt/{args.model}/{args.data}', exist_ok=True)
#     os.makedirs(f'log/{args.model}/{args.data}', exist_ok=True)

"""
    日志输出
"""
logger = logging.getLogger('mygo')
logger.setLevel(logging.INFO)
format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(format)
logger.addHandler(handler)
logger.info("Warning")

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
             visual_dropout=args.visual_dropout, textual_dropout=args.textual_dropout).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
