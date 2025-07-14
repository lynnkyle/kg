import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, n_feature, n_action, lr=0.001):
        """
        :param n_feature: 状态特征维度
        :param n_action: 动作数量
        :param lr:
        """
        super(Actor, self).__init__()
        self.n_action = n_action

        self.fc1 = nn.Linear(n_feature, n_feature)
        self.fc2 = nn.Linear(n_feature, n_action)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.fc2(x), dim=1)
        return probs

    def choose_action(self, state):
        """
        选择下一步动作
        :param state: 当前环境状态, 形状为(n_features,)
        :return: 选取的动作索引
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state).detach().numpy().flatten()  # 根据状态得到动作概率
        action = np.random.choice(self.n_action, p=probs)  # 根据概率采样动作
        return action

    def learn(self, state, action, td_error):
        """
        利用采样得到的动作和TD误差来更新Actor网络的参数，使网络更倾向于选择带来更高奖励的动作
        :param state: 当前状态, 形状为(n_features,)
        :param action: 采取的动作索引
        :param td_error: TD误差, 优势函数的估计值
        :return:
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.int64)
        td_error = torch.tensor([td_error], dtype=torch.float32)

        probs = self.forward(state)
        log_prob = torch.log(probs.squeeze(0)[action])
        loss = -log_prob * td_error

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Critic(nn.Module):
    def __init__(self, n_feature, lr=0.01):
        super(Critic, self).__init__()
        self.gamma = 0.9
        self.fc1 = nn.Linear(n_feature, n_feature)
        self.fc2 = nn.Linear(n_feature, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss_func = nn.MSELoss()

    def forward(self, x):
        """ 价值估值 """
        x = F.relu(self.fc1(x))
        v = self.fc2(x)
        return v

    def learn(self, s, r, s_):
        """
        :param s: 当前状态
        :param r: 即时奖励
        :param s_: 下一状态
        :return: TD误差
        """
        s = torch.FloatTensor(s).unsqueeze(0)
        s_ = torch.FloatTensor(s_).unsqueeze(0)
        r = torch.tensor([r], dtype=torch.float32)

        with torch.no_grad():
            v_next = self.forward(s_)
        v = self.forward(s)
        td_error = r + self.GAMMA * v_next - v
        loss = td_error.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return td_error.item()


def cosine_scores(emb1, emb2):
    emb1 = F.normalize(torch.tensor(emb1), dim=1)
    emb2 = F.normalize(torch.tensor(emb2), dim=1)
    return torch.matmul(emb1, emb2.t()).numpy()


if __name__ == '__main__':
    # 实体及其嵌入
    kg1_entities = ['a', 'b', 'c']
    kg2_entities = ['1', '2', '3']

    kg1_embeds = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.1], [0.6, 0.7, 0.8]])
    kg2_embeds = np.array([[0.11, 0.31, 0.49], [0.19, 0.38, 0.12], [0.58, 0.68, 0.82]])

    sim_scores = cosine_scores(kg1_embeds, kg2_embeds)
    print("相似度矩阵:\n", sim_scores)

    # 初始化 Actor-Critic
    n_feature = 3
    n_action = 3
    actor = Actor(n_feature, n_action, lr=0.001)
    critic = Critic(n_feature, lr=0.01)

    # 实验过程
    epochs = 20
    correct_total = []

    for ep in range(epochs):
        matched = {}
        available = set()
        acc = 0

        for i in range(len(kg1_entities)):
            state = sim_scores[i]
            mask = np.array([1 if j in available else 0 for j in range(3)])
            masked_state = state * mask
            action = actor.choose_action(masked_state)
            while action not in available:
                action = actor.choose_action(masked_state)
