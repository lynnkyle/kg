import os
import argparse

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms

"""
    RANK: 进程编号
    WORLD_SIZE: 进程总数
    SLURM_PROCID: 当前进程在作业中的唯一进程编号
"""


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


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
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                            rank=args.rank)
    dist.barrier(device_ids=[args.gpu])


"""
    分布式训练前, 进行软连接
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_path', type=str, default='/home/ps/lzy/vista/demo')
    args = parser.parse_args()
    init_distributed_mode(args)
    rank = args.rank
    batch_size = args.batch_size
    if rank == 0:
        print("集群初始化成功......")
    # 1.多gpu下划分数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='/home/ps/lzy/vista/demo/data/', transform=transform, train=True, download=True)
    test_data = datasets.MNIST(root="/home/ps/lzy/vista/demo/data/", transform=transform, train=False)
    mnist_dir = os.path.join(os.path.abspath(train_data.root), "MNIST")
    print("MNIST data is stored in:", mnist_dir)
    train_sample = torch.utils.data.distributed.DistributedSampler(train_data)
    test_sample = torch.utils.data.distributed.DistributedSampler(test_data)
    train_batch_sample = torch.utils.data.BatchSampler(train_sample, batch_size * 2, drop_last=True)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_sampler=train_batch_sample, num_workers=nw,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_sampler=test_sample, num_workers=nw,
                                              pin_memory=True)
    if rank == 0:
        print("数据集加载成功。")
    # 2.多gpu下加载模型
    save_path = args.save_path
    model = Model()
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'model.init.pth'))
