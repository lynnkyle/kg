from typing import Any

import math

import torch
from torch import nn
from torch.nn import functional as F

"""
    视觉虚拟嵌入生成器
"""


class VirEmbGen(nn.Module):
    def __init__(self, args, num_modal=3):
        super().__init__()
        self.args = args
        self.num_modal = num_modal
        self.ent_dim = int(self.args.hidden_units.strip().split(',')[0])
        if num_modal > 3:
            num_modal = 5
        self.dim = [self.ent_dim, self.args.attr_dim, self.args.attr_dim, self.args.name_dim, self.args.txt_dim]
        self.fc = nn.Linear(sum(self.dim[:num_modal]), self.args.vis_dim)

    def forward(self, embs):
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        hyb_emb = torch.cat(embs, dim=1)
        return F.normalize(self.trans_fc(hyb_emb))


"""
    视觉虚拟嵌入生成器VAE
"""


class VirEmbGen_vae(nn.Module):
    def __init__(self, args, modal_num=3):
        super(VirEmbGen_vae, self).__init__()
        self.args = args
        self.modal_num = modal_num
        self.ent_dim = int(self.args.hidden_units.strip().split(',')[0])
        if modal_num > 3:
            modal_num = 5

        self.dim = [self.ent_dim, self.args.attr_dim, self.args.attr_dim, self.args.name_dim, self.args.txt_dim]
        hidden_list = [self.args.vis_dim]

        self.vae = VAE(sum(self.dim[:modal_num]), hidden_list, self.args.vis_dim)

    def forward(self, embs):
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        hyb_emb = F.normalize(torch.cat(embs, dim=1))
        hidden, hidden_norm, decoder_output, kl_div = self.vae(hyb_emb)
        return F.normalize(hidden), F.normalize(hidden_norm), F.normalize(decoder_output), kl_div, hyb_emb


class VAE(nn.Module):
    def __init__(self, emb_size, hid_size_list, mid_hid):
        super(VAE, self).__init__()
        self.emb_size = emb_size
        self.hid_size_list = hid_size_list
        self.mid_hid = mid_hid
        self.enc_feat_size_list = [self.emb_size] + self.hid_size_list + [self.mid_hid * 2]
        self.dec_feat_size_list = [self.emb_size] + self.hid_size_list + [self.mid_hid]
        self.encoder = nn.ModuleList([
            nn.Linear(self.enc_feat_size_list[i], self.enc_feat_size_list[i + 1])
            for i in range(len(self.enc_feat_size_list) - 1)
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(self.dec_feat_size_list[i], self.dec_feat_size_list[i - 1])
            for i in range(len(self.dec_feat_size_list) - 1, 0, -1)
        ])

    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            x = self.encoder[i](x)
            if i != len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decode(self, x):
        for i, layer in enumerate(self.decoder):
            x = self.decoder[i](x)
            x = F.relu(x)
        return x

    def forward(self, x):
        encoder_output = self.encoder(x)
        mu, sigma = encoder_output.chunk(2, 1)  # mu , log_var
        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        hidden_norm = mu * torch.exp(sigma) ** 0.5  # var => std
        # hidden = torch.randn_like(sigma) * torch.exp(sigma) ** 0.5 + mu
        # hidden_norm = torch.exp(sigma) ** 0.5
        decoder_output = self.decoder(hidden)
        # kl散度计算公式 kl越小表示分布越相似, kl越大表示分布越不相似
        kl_div = 0.5 * torch.sum(torch.exp(sigma) + mu - sigma - 1) / (x.shape[0] * x.shape[1])
        return hidden, hidden_norm, decoder_output, kl_div


class GCN(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_feat, hid_feat)
        self.gc2 = GraphConvolution(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.intial_parameters()

    def intial_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GAT(nn.Module):
    def __init__(self, n_head, n_unit, dropout, attn_dropout, norm, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_unit) - 1
        self.dropout = dropout
        self.norm = norm
        if self.norm:
            self.inst_norm = nn.InstanceNorm1d(num_features=n_unit, momentum=0.0, affine=True)
        self.layers = nn.ModuleList()
        for i in range(self.num_layer):
            feat_in = n_unit[i]
            self.layers.append(MultiHeadGraphAttention(n_head[i], feat_in, n_unit[i + 1], attn_dropout, diag))

    def forward(self, x, adj):
        """
        :param x:
        :param adj:
        :return: [num_ent, feat_in]
        """
        if self.norm:
            x = self.inst_norm(x)

        for i, gat_layer in enumerate(self.layers):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout)

            x = self.gat_layer(x, adj)

            if self.diag:
                x = x.mean(dim=0)

            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))

        if not self.diag:
            x = x.mean(dim=0)

        return x


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, feat_in, feat_out, attn_dropout, diag=True, bias=False):
        """
        :param n_head:
        :param feat_in: feat_in == feat_out
        :param feat_out: feat_out
        :param attn_dropout:
        :param diag:
        """
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        self.diag = diag
        if self.diag:
            self.weight = nn.Parameter(torch.FloatTensor(n_head, 1, feat_out))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(n_head, feat_in, feat_out))
        self.attn = nn.Parameter(torch.FloatTensor(n_head, feat_out * 2, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(feat_out))

    def forward(self, x, adj):
        """
        :param x:   (num_ent, feat_in)
        :param adj: (num_edge, num_edge)
        :return:    (2, num_ent, feat_out)
        """
        output = []
        N = x.size()[0]
        edge = adj._indices()
        for i in range(self.n_head):
            if self.diag:
                h = torch.mul(x, self.weight[i])  # [num_ent, feat_out]
            else:
                h = torch.mm(x, self.weight[i])  # [num_ent, feat_out]
            edge_h = torch.cat((h[edge[0], :], h[edge[1], :]), dim=1)  # [num_ent, 2 * feat_out]
            edge_e = torch.exp(-self.leaky_relu(torch.mm(edge_h, self.attn[i]).squeeze()))  # [num_ent,]
            e_row_sum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones([N, 1]))  # [N, 1]
            edge_e = self.attn_dropout(edge_e)
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)  # [N, feat_out]
            h_prime = torch.div(h_prime, e_row_sum)  # [N, feat_out]
            output.append(h_prime.unsqueeze(0))  # list:2 [1, N, feat_out]
        output = torch.cat(output, dim=0)  # [2, N, feat_out]
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(torch.nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction(indices, values, shape, b)
