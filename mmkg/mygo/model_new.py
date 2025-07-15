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


class BrayCurtisAlign(nn.Module):
    def __init__(self):
        super(BrayCurtisAlign, self).__init__()

    def forward(self, emb_1, emb_2):
        emb_u = emb_1.unsqueeze(1)
        emb_v = emb_2.unsqueeze(0)
        numerator = torch.abs(emb_u - emb_v).sum(dim=2)
        denominator = torch.abs(emb_u + emb_v).sum(dim=2) + 1e-8
        d_bray = numerator / denominator
        return 1.0 - d_bray


class CosineAlign(nn.Module):
    def __init__(self):
        super(CosineAlign, self).__init__()

    def forward(self, emb1, emb2):
        u = F.normalize(emb1, p=2, dim=1)  # [n, d]
        v = F.normalize(emb2, p=2, dim=1)  # [m, d]
        # 执行矩阵乘法，相当于逐对做 dot product
        sim_matrix = torch.matmul(u, v.T)  # [n, m]
        return sim_matrix


class ICLLoss(nn.Module):
    def __init__(self):
        super(ICLLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.align_fn_1 = BrayCurtisAlign()
        self.align_fn_2 = CosineAlign()

    def forward(self, emb1, emb2):
        """:
        :param emb1: [batch_size, dim]->[batch_size, 1, dim]
        :param emb2: [batch_size, dim]->[1, batch_size, dim]
        """
        batch_sim_1 = self.align_fn_1(emb1, emb2)
        batch_sim_2 = self.align_fn_2(emb1, emb2)  # [batch_size, batch_size]
        """
            计算第一次、第二次过实体编码器Transformer得到的嵌入的损失
        """
        labels = torch.arange(batch_sim_2.size(0)).long().to('cuda')
        return self.loss(batch_sim_2, labels.cuda())


if __name__ == '__main__':
    x = torch.randn((3, 6, 5))
    print(x.size())
    print(x.size(0))
    print(x.size(1))
    print(x.size(2))
