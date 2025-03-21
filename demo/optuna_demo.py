import sys
import argparse
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=30)
args = parser.parse_args()


# 模型
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
            nn.MaxPool2d(kernel_size=2, stride=1)  # [batch, 128, 14, 14]
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


# 模型训练
def train_one_epoch(train_dataloader, model, optimizer, device):
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for step, data in enumerate(tqdm(train_dataloader, file=sys.stdout)):
        img, label = data
        optimizer.zero_grad()
        loss = loss_fn(model(img.to(device)), label.to(device))
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def get_metrics(test_loader, model, device):
    length = len(test_loader)
    total_num = 0
    with torch.no_grad():
        model.eval()
        for step, data in enumerate(tqdm(test_loader)):
            img, label = data
            pred = torch.max(model(img.to(device)), dim=1)[1]
            total_num += pred.eq(label.to(device)).sum().item()
    return {"acc": total_num / length}


# optuna调参
def adjust_parameter(trial):
    # max_epoch = trial.suggest_int('max_epoch', 30, 50)
    padding = trial.suggest_int('padding', 1, 5, step=1)
    kernel_size = trial.suggest_int('kernel_size', 1, 5, step=1)
    hid_dim = trial.suggest_int('hid_dim', 512, 4096)
    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    # 数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True)
    # 模型要素
    width = train_dataset[0][0].shape[1]
    high = train_dataset[0][0].shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(width, high, padding=padding, kernel_size=kernel_size, hid_dim=hid_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    # for epoch in range(max_epoch):
    loss = train_one_epoch(train_loader, model, optimizer, device)
    # print("Epoch {}, Loss: {}".format(epoch, loss))
    lr_scheduler.step()
    # if epoch % 5 == 0:
    acc = get_metrics(test_loader, model, device)
    # print(acc)
    return acc['acc']


if __name__ == '__main__':

    # # 数据集
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, shuffle=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # 模型要素
    # model = Model(3, 1024).to(device)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    #
    # for epoch in range(args.max_epoch):
    #     loss = train_one_epoch(train_loader, optimizer)
    #     print("Epoch {}, Loss: {}".format(epoch, loss))
    #     lr_scheduler.step()
    #     if epoch % 5 == 0:
    #         acc = get_metrics(test_loader)
    #         print(acc)

    study = optuna.create_study(direction='maximize')
    study.optimize(adjust_parameter, n_trials=3)
    print("Best Trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    for k, v in trial.params.items():
        print(f"{k}：{v}")
