from typing import Any

import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.pytorch_utils import apply_chunking_to_forward


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


"""
    Transformer中, Dropout之后再LayerNorm的顺序, 会使得数据更稳定
"""


class BertSelfAttention(nn.Module):
    """
        多头: 多对q k v 参数
    """

    def __init__(self, num_attn_head, hidden_size, dropout=0.1):
        super().__init__()
        self.num_attn_head = num_attn_head
        assert hidden_size % num_attn_head == 0
        self.hidden_size = hidden_size // self.num_head
        self.attn_head_size = int(hidden_size / num_attn_head)
        self.all_head_size = self.num_head * self.attn_head_size
        self.q = nn.Linear(self.hidden_size, self.all_head_size)
        self.k = nn.Linear(self.hidden_size, self.all_head_size)
        self.v = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state, output_attention=False):
        """
        :param hidden_state:    [batch_size, seq_len, hidden_size]
        :param output_attention:
        :return:    [batch_size, seq_len, all_head_size]
        """
        query_layer = self.multi_transpose(self.q(hidden_state))  # [batch_size, num_attn_head, seq_len, attn_head_size]
        key_layer = self.multi_transpose(self.k(hidden_state))  # [batch_size, num_attn_head, seq_len, attn_head_size]
        value_layer = self.multi_transpose(self.v(hidden_state))  # [batch_size, num_attn_head, seq_len, attn_head_size]
        attn_weight = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [batch_size, num_attn_head, seq_len, seq_len]
        attn_weight = attn_weight / math.sqrt(self.attn_head_size)
        # [batch_size, num_attn_head, seq_len, seq_len]
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
        # [batch_size, num_attn_head, seq_len, seq_len]
        attn_weight_dp = self.dropout(attn_weight)
        # [batch_size, num_attn_head, seq_len, seq_len]
        context_layer = torch.matmul(attn_weight_dp, value_layer)
        # [batch_size, num_attn_head, seq_len, attn_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # [batch_size, seq_len, num_attn_head, attn_head_size]
        context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [batch_size, seq_len, all_head_size]
        context_layer = context_layer.view(context_layer_shape)
        # [batch_size, seq_len, all_head_size]
        if output_attention:
            return (context_layer, attn_weight)
        else:
            return (context_layer,)

    def multi_transpose(self, x):
        """
        :param x: [batch_size, seq_len, hidden_size]
        :return:
        """
        shape = x.size()[:-1] + (self.num_attn_head, self.attn_head_size)
        x = x.view(shape)  # [batch_size, seq_len, num_attn_head, attn_head_size]
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_attn_head, seq_len, attn_head_size]
        return x


class BerSelfOutput(nn.Module):
    """
        自注意力投影层
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state, x):
        hidden_state = self.linear(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layer_norm(x + hidden_state)
        return hidden_state


class BertAttention(nn.Module):
    def __init__(self, num_attn_head, hidden_size, dropout=0.1):
        super().__init__()
        self.self_attn = BertSelfAttention(num_attn_head=num_attn_head, hidden_size=hidden_size, dropout=dropout)
        self.self_output = BerSelfOutput(hidden_size=hidden_size, dropout=dropout)

    def forward(self, x, output_attention=False):
        self_attn_output = self.self_attn(x, output_attention)
        self_out_output = self.self_output(self_attn_output[0], x)
        output = (self_out_output,) + self_attn_output[1:]
        return output


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        return x


class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state, x):
        hidden_state = self.linear(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layer_norm(x + hidden_state)
        return hidden_state


class BertLayer(nn.Module):
    def __init__(self, num_attn_head, hidden_size, intermediate_size, dropout=0.1, use_intermediate=True):
        super().__init__()
        self.seq_len_dim = 1  # 指定seq_len的维度
        self.chunk_size_feed_forward = 0
        self.bert_attn = BertAttention(num_attn_head=num_attn_head, hidden_size=hidden_size, dropout=dropout)
        self.use_intermediate = use_intermediate
        if use_intermediate:
            self.intermediate = BertIntermediate(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.output = BertOutput(hidden_size=hidden_size, intermediate_size=intermediate_size, dropout=dropout)

    def forward(self, x, output_attention=False):
        self_attn_output = self.bert_attn(x, output_attention)
        if not self.use_intermediate:
            return (self_attn_output[0], self_attn_output[1])
        attn_output = self_attn_output[0]
        attn_weight = self_attn_output[1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        output = (layer_output, attn_weight)
        return output

    def feed_forward_chunk(self, x):
        hidden_state = self.intermediate(x)
        output = self.output(hidden_state, x)
        return output
