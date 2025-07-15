import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, n_feature, n_action, lr=0.001):
        super(Actor, self).__init__()
        self.n_action = n_action
        self.fc1 = nn.Linear(n_feature, 1024)
        self.fc2 = nn.Linear(1024, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.fc2(x), dim=1)
        return probs

    def choose_action(self, state, available=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state).detach().numpy().flatten()

        if available is not None:
            mask = np.zeros_like(probs)
            mask[list(available)] = 1
            probs = probs * mask
            if probs.sum() == 0:
                probs[list(available)] = 1.0
            probs = probs / probs.sum()

        action = np.random.choice(self.n_action, p=probs)
        return action

    def learn(self, state, action, td_error):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.int64)
        td_error = torch.tensor([td_error], dtype=torch.float32)

        probs = self.forward(state)
        log_prob = torch.log(probs.squeeze(0)[action])
        loss = -log_prob * td_error
        return loss


class Critic(nn.Module):
    def __init__(self, n_feature, lr=0.01):
        super(Critic, self).__init__()
        self.gamma = 0.9
        self.fc1 = nn.Linear(n_feature, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc2(x)
        return v

    def learn(self, s, r, s_):
        s = torch.FloatTensor(s).unsqueeze(0)
        s_ = torch.FloatTensor(s_).unsqueeze(0)
        r = torch.tensor([r], dtype=torch.float32)

        with torch.no_grad():
            v_next = self.forward(s_)
        v = self.forward(s)
        td_error = r + self.gamma * v_next - v
        loss = td_error.pow(2).mean()
        return loss, td_error.detach()


def cosine_scores(emb1, emb2):
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
    return torch.matmul(emb1, emb2.t())


def contrastive_loss(sim_matrix):
    batch_size = sim_matrix.size(0)
    pos = torch.diag(sim_matrix)
    neg = sim_matrix[~torch.eye(batch_size, dtype=bool)].view(batch_size, -1)
    loss = -torch.log(pos / (pos + neg.sum(dim=1) + 1e-8)).mean()
    return loss


def train_actor_critic(kg1_embeds, kg2_embeds, actor, critic, optimizer,
                       epochs=50, patience=3, n_action=3):
    patience_counter = 0
    best_acc = 0
    correct_total = []

    for ep in range(epochs):
        matched = {}
        available = set(range(n_action))
        acc = 0
        sim_scores = cosine_scores(kg1_embeds, kg2_embeds)
        loss_contrast = contrastive_loss(sim_scores)

        total_actor_loss = 0
        total_critic_loss = 0

        for i in range(n_action):
            state = sim_scores[i]
            mask = torch.tensor([1 if j in available else 0 for j in range(n_action)], dtype=torch.float32)
            masked_state = state * mask

            action = actor.choose_action(masked_state.detach().numpy(), available=available)
            reward = 1.0 if i == action else 0.0
            available.remove(action)
            matched[i] = action
            acc += reward

            if i < n_action - 1:
                next_state = sim_scores[i + 1]
                next_mask = torch.tensor([1 if j in available else 0 for j in range(n_action)], dtype=torch.float32)
                masked_next_state = next_state * next_mask
            else:
                masked_next_state = torch.zeros_like(state)

            critic_loss, td_error = critic.learn(state.detach().numpy(), reward, masked_next_state.detach().numpy())
            actor_loss = actor.learn(state.detach().numpy(), action, td_error)

            total_critic_loss += critic_loss
            total_actor_loss += actor_loss

        optimizer.zero_grad()
        (loss_contrast + total_actor_loss + total_critic_loss).backward()
        optimizer.step()

        acc_ratio = acc / n_action
        correct_total.append(acc_ratio)
        print(f"[Episode {ep}] 准确率: {acc_ratio:.3f}, 匹配: {matched}, 对比损失: {loss_contrast.item():.4f}")

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    print(f"平均准确率: {np.mean(correct_total):.4f}")
    return correct_total, best_acc


if __name__ == '__main__':
    kg1_embeds = nn.Parameter(torch.tensor([[0.1, 0.3, 0.5],
                                            [0.2, 0.4, 0.1],
                                            [0.6, 0.7, 0.8]], dtype=torch.float32))
    kg2_embeds = nn.Parameter(torch.tensor([[0.11, 0.31, 0.49],
                                            [0.19, 0.38, 0.12],
                                            [0.58, 0.68, 0.82]], dtype=torch.float32))
    n_feature = 3
    n_action = 3
    actor = Actor(n_feature, n_action)
    critic = Critic(n_feature)

    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [kg1_embeds, kg2_embeds], lr=0.001)

    train_actor_critic(kg1_embeds, kg2_embeds, actor, critic, optimizer,
                       epochs=50, patience=3, n_action=n_action)
