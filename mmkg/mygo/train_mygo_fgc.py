import os
import sys
import argparse

import torch
import logging

from dataset import VTKG

"""
    参数设置
"""
parser = argparse.ArgumentParser()
parser.add_argument("--no_write")
args = parser.parse_args()
"""
    文件保存
"""
# if not args.no_write:
#     os.makedirs(f'ckpt/{args}')
#     os.makedirs()
#     os.makedirs()

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
dataset = VTKG(data='MKG-W', max_vis_len=-1)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
print(data_loader)
