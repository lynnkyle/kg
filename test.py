import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from kg.rotate import RotatE

# 随机种子的设置
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# configuration
device = torch.device("cuda:0")

# 1. 创建数据集
triples = np.random.randint(low=0, high=10, size=(10000, 3))
print(triples)


def objective(trial):
    emb_dim = trial.suggest_int('emb_dim', 100, 10000)
    gamma = trial.suggest_int('gamma', 100, 10000)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    # 2. 创建模型
    model = RotatE(num_ent=10, num_rel=10, emb_dim=emb_dim, device=device, gamma=gamma)
    model.to(device=device)
    # 3. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 4. 模型训练
    max_epoch = 11
    log_step = 1000
    test_step = 100
    losses = []
    batch_size = 100
    num_triple = triples.shape[0]
    n_batch = num_triple // batch_size
    ls = random.sample(range(n_batch), n_batch)
    # print(len(ls))
    # 迭代次数 5001*100
    for epoch in range(max_epoch):
        model.train()
        for idx in ls:
            start = idx * batch_size
            end = min((idx + 1) * batch_size, num_triple)
            loss = model.get_loss(triples[start:end])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # if epoch % log_step == 0:
        #     print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
        # if epoch % test_step == 0:
        #     metrics = model_0.get_metrics(triples)
        #     print(metrics)
    return loss.item()


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)
print("Best Trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
for key, value in trial.params.items():
    print(f"    {key}:{value}")
