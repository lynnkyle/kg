import os
import argparse

import torch
import torch.distributed as dist

"""
    RANK: 进程编号
    WORLD_SIZE: 进程总数
    SLURM_PROCID: 当前进程在作业中的唯一进程编号
"""


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'


"""
    分布式训练前, 进行软连接
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    print("rank==>", args.rank)
    print("world_size==>", args.world_size)
    print("gpu==>", args.gpu)
