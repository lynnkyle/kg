import torch
from torch import nn
import torch.nn.functional as F

"""
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" CVPR 2018
    (MLE)  [由最大似然估计推导得出的损失函数] [非标准版]
"""


class CustomMultiLossLayer(nn.Module):
    def __init__(self, loss_num):
        super().__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) <= self.loss_num
        precisions = torch.exp(-self.log_vars)
        loss = 0
        for i in range(len(loss_list)):
            loss += loss_list[i] * precisions[i] + self.log_vars[i]
        return loss


"""
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" CVPR 2018
    (MLE)  [由最大似然估计推导得出的损失函数] [标准版]
"""


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, loss_num):
        super().__init__()
        self.loss_num = loss_num
        self.params = nn.Parameter(torch.ones(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) <= self.loss_num
        loss = 0
        for i in range(len(loss_list)):
            loss += 0.5 / (self.params[i] ** 2) * loss_list[i] + torch.log(1 + self.params[i] ** 2)
        return loss


"""
    损失函数基于对比学习, 主要思想是:
    1. 相似实体之间的嵌入相近
    2. 不同实体之间的嵌入相对远离
    3. 通过温度参数来控制相似度的分布
"""


def cosine_sim(im, s):
    return im.mm(s.t())


class IclLoss(nn.Module):
    def __init__(self, tau=0.05, ab_weight=0.5, n_view=2, intra_weight=1.0, inversion=False, neg_cross_kg=False):
        super().__init__()
        self.tau = tau
        self.sim = cosine_sim
        self.weight = ab_weight
        self.n_view = n_view
        self.intra_weight = intra_weight
        self.inversion = inversion
        self.neg_cross_kg = neg_cross_kg

    def forward(self, emb, train_links, neg_l=None, neg_r=None, weight_norm=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
        zis = self.emb[train_links[:, 0]]  # embedding for i-th samples
        zjs = self.emb[train_links[:, 1]]  # embedding for i-th samples
        if weight_norm is not None:
            zis_w = weight_norm[train_links[:, 0]]
            zjs_w = weight_norm[train_links[:, 1]]
            score_w = torch.stack([zis_w, zjs_w], dim=1)
            score_w_min = torch.min(score_w, 1)[0]
        else:
            score_w_min = None

        temperature = self.tau
        alpha = self.weight
        n_view = self.n_view
        LARGE_NUM = 1e9  # 对角线掩码
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]
        hidden1_large = hidden1
        hidden2_large = hidden2

        if neg_l is None:
            num_classes_1 = batch_size * n_view
        else:
            num_classes_1 = batch_size * n_view + neg_l.shape[0]
            num_classes_2 = batch_size * n_view + neg_r.shape[0]

        labels = F.one_hot().cuda()  # 创建一个one-hot编码的标签矩阵，用于对比学习的损失计算
