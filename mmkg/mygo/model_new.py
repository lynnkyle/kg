import torch
import torch.nn.functional as F
from torch import nn


class Tucker(nn.Module):
    def __init__(self, e_dim, r_dim):
        super(Tucker, self).__init__()
        # self.e_dim = e_dim
        # self.r_dim = r_dim
        self.W = nn.Parameter(torch.rand(r_dim, e_dim, e_dim))  # [r_dim, e_dim, e_dim]
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(e_dim)
        self.bn1 = nn.BatchNorm1d(e_dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.output_drop = nn.Dropout(0.5)

    """
        计算的是: W * e * r, 计算的不是得分
    """

    def forward(self, ent_emb, rel_emb):
        """
        :param ent_emb: [batch_size, e_dim]
        :param rel_emb: [batch_size, r_dim]
        :return: [batch_size, e_dim]
        """
        x = self.bn0(ent_emb)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))  # [batch_size, 1, e_dim]

        r = torch.mm(rel_emb, self.W.view(rel_emb.size(1), -1))  # [batch_size, e_dim * e_dim]
        r = r.view(-1, x.size(2), x.size(2))  # [batch_size, e_dim, e_dim]
        r = self.hidden_drop(r)

        x = torch.bmm(x, r)  # [batch_size, 1, e_dim]
        x = x.view(-1, x.size(2))  # [batch_size, e_dim]
        x = self.bn1(x)
        x = self.output_drop(x)
        return x


"""
    计算嵌入的相似度
"""


class ICLLoss(nn.Module):
    def __init__(self, temp=0.05):
        super(ICLLoss, self).__init__()
        self.temp = temp
        self.LARGE_NUM = 1e9

    def forward(self, emb1, emb2):
        """:
        :param emb1: [batch_size, dim]->[batch_size, 1, dim]
        :param emb2: [batch_size, dim]->[1, batch_size, dim]
        """

        """
            计算第一次、第二次过实体编码器Transformer得到的嵌入的损失
        """
        assert emb1.size() == emb2.size()
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        ent_num = emb1.shape[0]
        target = F.one_hot(torch.arange(0, ent_num), num_classes=ent_num * 2)
        mask = F.one_hot(torch.arange(ent_num), num_classes=ent_num)
        logits_aa = torch.matmul(emb1, emb1.t()) / self.temp - self.LARGE_NUM * mask
        logits_ab = torch.matmul(emb1, emb2.t()) / self.temp - self.LARGE_NUM * mask
        logits_bb = torch.matmul(emb2, emb2.t()) / self.temp - self.LARGE_NUM * mask
        logits_ba = torch.matmul(emb2, emb1.t()) / self.temp - self.LARGE_NUM * mask
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        loss = self.softXEnt(target, logits_a)
        return loss

    def softXEnt(self, target, logits):
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(target * log_probs).sum() / logits.shape[0]
        return loss


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        batch_sim = self.sim_func(emb1.unsqueeze(1), emb2.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)


if __name__ == '__main__':
    x = torch.randn((5, 256))
    y = torch.randn((5, 256))
    loss_func = ICLLoss(temp=0.05)
    loss = loss_func(x, y)
