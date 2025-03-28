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
    [!!!important]
    ICL对比学习损失函数
    核心思想: 让正样本更相似,负样本更不相似
    计算方式: 样本的相似度矩阵，并使用交叉熵损失进行对比学习
"""


class IclLoss(nn.Module):
    def __init__(self, tau=0.05, modal_weight=0.5, n_view=2, intra_weight=1.0, inversion=False):
        super().__init__()
        self.tau = tau
        self.modal_weight = modal_weight
        self.n_view = n_view
        self.intra_weight = intra_weight
        self.inversion = inversion

    def forward(self, emb, train_links, neg_l=None, neg_r=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
        zis = emb[train_links[:, 0]]  # embedding for i-th samples
        zjs = emb[train_links[:, 1]]  # embedding for j-th samples

        score_w_min = None
        temperature = self.tau
        alpha = self.modal_weight
        n_view = self.n_view
        LARGE_NUM = 1e9  # 对角线掩码
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]
        hidden1_large = hidden1
        hidden2_large = hidden2
        num_classes_1 = batch_size * n_view
        if neg_l is not None:
            num_classes_1 = batch_size * n_view + neg_l.shape[0]
            num_classes_2 = batch_size * n_view + neg_r.shape[0]
            labels_2 = F.one_hot(torch.arange(start=0, end=batch_size), num_classes=num_classes_2).float().cuda()
        labels_1 = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64),
                             num_classes=num_classes_1).float().cuda()  # 创建一个one-hot编码的标签矩阵，用于对比学习的损失计算
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64),
                          num_classes=batch_size).float().cuda()

        """
            计算所有样本的点积相似性
            logits_aa, logits_bb主要用于构造负样本, F.softmax中会使用到
        """
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM

        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM

        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        loss_a = self.softXEnt(labels_1, logits_a, w_min=score_w_min)
        loss_b = self.softXEnt(labels_1, logits_b, w_min=score_w_min)

        return alpha * loss_a + (1 - alpha) * loss_b

    def softXEnt(self, target, logits, w_min=None):
        """
        :param target:  [batch_size, n_view]
        :param logits:  [batch_size, n_view]
        :param w_min:
        :return:
        """
        log_probs = F.log_softmax(logits, dim=1)
        if w_min is None:
            w_min = w_min.unsqueeze(1)
            loss = - (target * log_probs * w_min).sum() / logits.shape[0]
        else:
            loss = - (target * log_probs).sum() / logits.shape[0]
        return loss


"""
    [!!!important]
    IAL对比学习损失函数
    核心思想: 最小化KL散度, 让不同模态的表示对齐
    计算方式: 让单模态和多模态的相似性分布，并使用KL散度约束
"""


class IalLoss(nn.Module):
    def __init__(self, tau=0.05, modal_weight=0.5, n_view=2, zoom=1, inversion=False, reduction='mean', detach=False):
        super().__init__()
        self.tau = tau
        self.modal_weight = modal_weight
        self.n_view = n_view
        self.zoom = zoom
        self.inversion = inversion
        self.reduction = reduction
        self.detach = detach

    def forward(self, src_emb, tar_emb, train_links, norm=True):
        if norm:
            src_emb = F.normalize(src_emb, dim=1)
            tar_emb = F.normalize(tar_emb, dim=1)

        src_zis = src_emb[train_links[:, 0]]
        src_zjs = src_emb[train_links[:, 1]]
        tar_zis = tar_emb[train_links[:, 0]]
        tar_zjs = tar_emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.modal_weight

        assert src_zis.shape[0] == tar_zjs.shape[0]

