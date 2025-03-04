import os
import argparse
import sys

from tqdm import tqdm
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
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # [batch_size, 64, 28, 28]
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [batch_size, 128, 28, 28]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),  # [batch_size, 128, 14, 14]
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),  # [batch_size, 1024]
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)  # [batch_size, 10]
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


def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


"""
    模型训练一轮
"""


def train_one_epoch(model, optimizer, data_loader, loss_fn, device, epoch):
    mean_loss = torch.zeros(1).to(device)
    model.train()
    if rank == 0:
        data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        x, label = data
        pred = model(x.to(device))
        optimizer.zero_grad()
        loss = loss_fn(pred, label.to(device))
        loss.backward()
        loss = reduce_value(loss)
        mean_loss = (mean_loss * step + loss.item()) / (step + 1)
        optimizer.step()
        optimizer.zero_grad()
        # # 在进程0中打印平均loss
        # if rank == 0:
        #     print("[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3)))
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    return mean_loss.item()


"""
    模型评估
"""


def evaluate(model, data_loader, device):
    with torch.no_grad():
        model.eval()
        # 用于存储预测正确的样本数量
        sum_num = torch.zeros(1).to(device)
        if rank == 0:
            data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            x, label = data
            pred = model(x.to(device))
            pred = torch.max(pred, dim=1)[1]
            sum_num += (pred == label.to(device)).sum()
        sum_num = reduce_value(sum_num)
        if device != torch.device('cpu'):
            torch.cuda.synchronize(device)
        return {'sum_num': sum_num}


"""
    分布式训练前, 进行软连接
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--model_init_path', type=str, default='/home/ps/lzy/vista/demo/model.init.pth')
    parser.add_argument('--save_path', type=str, default='/home/ps/lzy/vista/demo')
    parser.add_argument('--device', default='cuda', help='device id')
    args = parser.parse_args()
    init_distributed_mode(args)
    rank = args.rank
    batch_size = args.batch_size
    device = args.device
    if rank == 0:
        print("集群初始化成功......")
    # 1.多gpu下划分数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='/home/ps/lzy/vista/demo/data/', transform=transform, train=True, download=True)
    test_data = datasets.MNIST(root="/home/ps/lzy/vista/demo/data/", transform=transform, train=False)
    mnist_dir = os.path.join(os.path.abspath(train_data.root), "MNIST")
    print("MNIST data is stored in:", mnist_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_sampler=train_batch_sampler, num_workers=nw,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, sampler=test_sampler, num_workers=nw,
                                              pin_memory=True)
    if rank == 0:
        print("数据集加载成功。")
    # 2.多gpu下加载模型
    save_path = args.save_path
    model = Model().to(device)
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'model.init.pth'))
    dist.barrier(device_ids=[args.gpu])
    model.load_state_dict(torch.load(args.model_init_path, map_location=device))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    if rank == 0:
        print('模型加载成功。')
    # 3.模型要素
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    # 4.多gpu下模型训练(损失函数)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    loss_fn=loss_fn,
                                    device=device,
                                    epoch=epoch)
        lr_scheduler.step()
        sum_num = evaluate(model, test_loader, device)
        if rank == 0:
            print("epoch:{},mean_loss:{}".format(epoch, mean_loss))
            print("acc:{}".format(sum_num))

    if rank == 0:
        print("模型训练结束")
