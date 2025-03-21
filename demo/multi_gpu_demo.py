import os
import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, BatchSampler, DataLoader
from torch.utils.data._utils.collate import default_collate
import torch.distributed as dist
from torchvision import datasets, transforms

"""
    分布式初始化
"""


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                            rank=args.rank)
    dist.barrier(device_ids=[args.gpu])


def reduce_value(value, args, average=True):
    world_size = args.world_size
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


"""
    模型
"""


class Model(nn.Module):
    def __init__(self, hid_dim):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            # [batch, 64, 28, 28]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            # [batch, 128, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [batch, 128, 14, 14]
        )
        self.decoder = nn.Sequential(
            nn.Linear(128 * 14 * 14, hid_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hid_dim, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.decoder(x)
        return x


"""
    模型训练
"""


def train_one_epoch(data_loader, model, optimizer, args):
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    model.train()
    if rank == 0:
        data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        img, label = data
        pred = model(img.to(args.device))
        optimizer.zero_grad()
        loss = loss_fn(pred, label.to(args.device))
        loss.backward()
        loss = reduce_value(loss, args)
        optimizer.step()
        total_loss += loss.item()
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    return total_loss


def valid_eval_metrics(data_loader, model, args):
    positive_num = torch.zeros(1).to(args.device)
    with torch.no_grad():
        model.eval()
        if rank == 0:
            data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):
            img, label = data
            pred = torch.max(model(img.to(device)), dim=1)[1]
            positive_num += (pred == label.to(device)).sum()
    positive_num = reduce_value(positive_num, args)
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    return {'acc': positive_num / len(data_loader)}


if __name__ == '__main__':

    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--root_path', default='/home/ps/lzy/vista/demo', type=str)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    init_distributed_mode(args)
    rank = args.rank
    world_size = args.world_size
    root_path = args.root_path
    batch_size = args.batch_size
    device = args.device
    if rank == 0:
        print("分布式环境初始化成功")

    # 数据集加载
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=os.path.join(root_path, 'data'), train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=os.path.join(root_path, 'data'), train=False, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    train_batch_sampler = BatchSampler(train_sampler, batch_size=batch_size, drop_last=True)
    nw = min([os.cpu_count(), batch_size if batch_size > 0 else 0, 8])
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=nw, pin_memory=True,
                              collate_fn=default_collate)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, num_workers=nw, pin_memory=True,
                             collate_fn=default_collate)

    # 模型要素
    model = Model(hid_dim=2048).to(device=device)
    if rank == 0:
        torch.save({'state_dict': model.state_dict()}, os.path.join(root_path, 'model_init.pth'))
    dist.barrier(device_ids=[args.gpu])
    model.load_state_dict(torch.load(os.path.join(root_path, 'model_init.pth'), map_location=device)['state_dict'])
    model = DistributedDataParallel(model, device_ids=[args.gpu])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # 模型训练
    for epoch in range(args.max_epochs):
        train_sampler.set_epoch(epoch)
        tatal_loss = train_one_epoch(train_loader, model, optimizer, args)
        acc = valid_eval_metrics(test_loader, model, args)['acc']
        if rank == 0:
            print(f"epoch {epoch} train loss: {tatal_loss}")
            print(f"epoch {epoch} test acc: {acc}")
        lr_scheduler.step()

    if rank == 0:
        if os.path.exists(os.path.join(root_path, 'model_init.pth')):
            os.remove(os.path.join(root_path, 'model_init.pth'))
