import sys
import argparse

import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('', )
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=30)
args = parser.parse_args()

# 数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),  # [batch, 64, 28, 28]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),  # [batch, 128, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [batch, 128, 14, 14]
        )
        self.decoder = nn.Sequential(
            nn.Linear(128 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.decoder(x)
        return x


# 模型要素
model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)


def train_one_epoch(epoch, train_dataloader, ):
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for step, data in enumerate(tqdm(train_dataloader, file=sys.stdout)):
        img, label = data
        optimizer.zero_grad()
        loss = loss_fn(model(img.to(device)), label.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


# 模型训练
for epoch in range(args.max_epoch):
    loss = train_one_epoch(epoch, train_dataloader)
