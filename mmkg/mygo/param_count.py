import argparse

import torch
from mmkg.mygo.dataset import VTKG

"""
    参数设置
"""
torch.cuda.set_device(2)
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='MKG-W')
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--model', type=str, default='AFFT')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_epoch', type=int, default=1500)
parser.add_argument('--valid_epoch', type=int, default=1)
parser.add_argument('--str_dim', default=256, type=int)
parser.add_argument('--num_kernels', default=512, type=int)
parser.add_argument('--max_vis_token', default=8, type=int)
parser.add_argument('--max_txt_token', default=8, type=int)
parser.add_argument("--no_write", action='store_true')
parser.add_argument('--str_dropout', default=0.9, type=float)
parser.add_argument('--visual_dropout', default=0.4, type=float)
parser.add_argument('--textual_dropout', default=0.1, type=float)
parser.add_argument('--lr', default=5e-4, type=float)
# Loss的超参数
parser.add_argument('--align_former', default=True, action='store_true')
parser.add_argument('--contrastive', default=0.001, type=float)
parser.add_argument('--before_align', default=0.001, type=float)
parser.add_argument('--after_align', default=0.001, type=float)
# Transformer的配置
parser.add_argument('--num_head', default=4, type=int)
parser.add_argument('--dim_hid', default=1024, type=int)
parser.add_argument('--num_layer_enc_ent', default=1, type=int)
parser.add_argument('--num_layer_enc_rel', default=1, type=int)
parser.add_argument('--num_layer_dec', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
args = parser.parse_args()

"""
    创建数据集
"""
kg = VTKG(data=args.data, max_vis_len=-1)
kg_loader = torch.utils.data.DataLoader(kg, batch_size=args.batch_size, shuffle=False)
print(kg.num_ent)
print(kg.num_rel)
print(len(kg.train))
print(len(kg.valid))
print(len(kg.test))

"""
    计算稀疏度
"""
