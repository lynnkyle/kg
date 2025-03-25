from typing import Any

import math
import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, emb_size, hid_size_list, mid_hid):
        super().__init__()
        self.emb_size = emb_size
        self.hid_size_list = hid_size_list
        self.mid_hid = mid_hid
        self.enc_feat_size_list = [emb_size] + self.hid_size_list + [self.mid_hid * 2]
        self.dec_feat_size_list = [emb_size] + self.hid_size_list + [self.mid_hid]
        self.encoder = nn.ModuleList(
            [nn.Linear(self.enc_feat_size_list[i], self.enc_feat_size_list[i + 1])
             for i in range(len(self.enc_feat_size_list) - 1)]
        )
        self.decoder = nn.ModuleList(
            [nn.Linear(self.dec_feat_size_list[i], self.dec_feat_size_list[i - 1])
             for i in range(len(self.dec_feat_size_list) - 1, 0, -1)]
        )

    def encoder(self, x):
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx != len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decoder(self, x):
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            x = F.relu(x)
        return x

    def forward(self, x):
        encoder_output = self.encoder(x)
        mean, sigma = encoder_output.chunk(2, dim=1)
        hidden = mean + torch.exp(sigma) ** 0.5 * torch.randn_like(sigma)
        hidden_norm = torch.exp(sigma) ** 0.5
        decoder_output = self.decoder(hidden)
        kl_div = 0.5 * torch.sum((torch.exp(sigma) + mean - sigma - 1)) / (x.shape[0] * x.shape[1])
        return hidden, hidden_norm, decoder_output, kl_div


class GraphConvolution(nn.Module):
    def __init__(self, in_feat, out_feat, bias=True):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feat))
        else:
            self.register_parameter('bias', None)
        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MultiHeadGraphAttention(nn.Module):
    """
        多头: 多对注意力参数, 多对线性变换参数
    """

    def __init__(self, n_head, in_feat, out_feat, attn_drop, diag=True, init=None, bias=False):
        super().__init__()
        self.n_head = n_head
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.attn_drop = attn_drop
        self.diag = diag
        self.attn = nn.Parameter(torch.Tensor(n_head, out_feat * 2, 1))  # 注意力参数 [n_head, out_feat * 2, 1]
        if self.diag:  # 逐元素缩放
            self.w = nn.Parameter(torch.FloatTensor(n_head, 1, out_feat))
        else:  # 线性变换
            self.w = nn.Parameter(torch.FloatTensor(n_head, in_feat, out_feat))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()

    def forward(self, x, adj):
        """
        :param x:   [ent_num, emb_dim]
        :param adj:     [ent_num, ent_num]
        emb_dim == in_feat
        :return:    [num_head, ent_num, out_feat]
        """
        output = []
        for i in range(self.n_head):
            N = x.size()[0]
            edge = adj._indices()
            if self.diag:
                h = torch.mul(x, self.w[i])  # [ent_num, emb_dim] [1, out_feat] => [ent_num, out_feat]
            else:
                h = torch.mm(x, self.w[i])  # [ent_num, emb_dim] [in_feat, out_feat] => [ent_num, out_feat]
            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1)  # [edge_num, out_feat * 2]
            edge_e = torch.exp(-self.leaky_relu(torch.mm(edge_h, self.attn[i])))
            # [edge_num, 1] [!!!important]计算的是每两个特征向量的权重值
            e_row_sum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)))
            # [2, edge_num] [edge_num,] [N, N], [N,1] => [ent_num, 1]
            edge_e = F.dropout(edge_e, self.attn_drop)
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            # [2, edge_num] [edge_num,] [N, N], [ent_num, out_feat] => [ent_num, out_feat]
            # 权重乘以特征向量
            h_prime = torch.div(h_prime, e_row_sum)  # [ent_num, out_feat]
            output.append(h_prime.unsqueeze(0))
        output = torch.cat(output, dim=0)  # [num_head, ent_num, out_feat]
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shapes, b):
        return SpecialSpmmFunction.apply(indices, values, shapes, b)


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shapes, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shapes)
        ctx.save_for_backward(a, b)
        ctx.N = shapes[0]
        return torch.matmul(a, b)  # matmul是矩阵乘法

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a = torch.matmul(grad_output, b.t())
            idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a.view(-1)[idx]
        if ctx.needs_input_grad[3]:
            grad_b = torch.matmul(grad_output, a.t())
        return None, grad_values, None, grad_b


class ProjectionHead(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(in_feat, hid_feat)
        self.linear_2 = nn.Linear(hid_feat, out_feat)
        self.dropout = dropout

    def forward(self, x):
        if x is not None:
            x = self.linear_1(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout)
            x = self.linear_2(x)
            return x


class BertSelfAttention(nn.Module):
    """
        多头: 多对q k v 参数
    """

    def __init__(self, num_attn_head, in_feat, out_feat):
        super().__init__()
        self.num_attn_head = num_attn_head
        assert in_feat % num_attn_head == 0
        self.in_feat = in_feat // self.num_head
        self.attn_head_size = int(in_feat / num_attn_head)
        self.all_head_size = self.num_head * self.attn_head_size
        self.q = nn.Linear(self.in_feat, self.all_head_size)
        self.k = nn.Linear(self.in_feat, self.all_head_size)
        self.v = nn.Linear(self.in_feat, self.all_head_size)

    def forward(self, hidden_state, output_attention=False):
        query_layer = self.q(hidden_state)
        key_layer = self.k(hidden_state)
        value_layer = self.v(hidden_state)

    def transpose(self, x):
        """
        :param x: [batch_size, seq_len, hidden_size]
        :return:
        """
        shape = x.size()[:-1] + (self.num_attn_head, self.attn_head_size)
        x = x.view(shape)  # [batch_size, seq_len, num_attn_head, attn_head_size]
        x = x.permute()
        return x
