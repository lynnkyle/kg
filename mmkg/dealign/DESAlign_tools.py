from typing import Any

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiModalEncoder(nn.Module):
    def __init__(self, args, ent_num, use_project_head=False,
                 name_feat_dim=300, attr_feat_dim=1000, rel_feat_dim=1000, vis_feat_dim=2048, txt_feat_dim=100):
        super(MultiModalEncoder, self).__init__()
        self.args = args
        attr_dim = self.args.attr_dim  # ???
        vis_dim = self.args.vis_dim
        txt_dim = self.args.txt_dim

        # Entity Embedding
        self.ent_num = ent_num
        self.ent_dim = int(self.args.hidden_units.strip().split(',')[0])
        self.ent_emb = nn.Embedding(ent_num, self.ent_dim)
        nn.init.normal_(self.ent_emb.weight, std=1.0 / math.sqrt(self.ent_num))
        self.ent_emb.requires_grad = True

        # Modal Encoder
        self.name_fc = nn.Linear(name_feat_dim, txt_dim)  # name_feat_dim
        self.att_fc = nn.Linear(attr_feat_dim, attr_dim)
        self.rel_fc = nn.Linear(rel_feat_dim, attr_dim)
        self.vis_fc = nn.Linear(vis_feat_dim, vis_dim)
        self.txt_fc = nn.Linear(txt_feat_dim, txt_dim)

        self.vir_emb_gen_vae = VirEmbGen_vae(args=self.args, modal_num=self.args.modal_num)
        if self.args.structure_encoder == 'gcn':
            pass
        elif self.args.structure_encoder == 'gat':
            pass


class VirEmbGen_vae(nn.Module):
    def __init__(self, args, modal_num=3):
        super(VirEmbGen_vae, self).__init__()
        self.args = args
        self.modal_num = modal_num
        self.ent_dim = int(self.args.hidden_units.strip().split(',')[0])
        if modal_num > 3:
            modal_num = 5

        self.dim = [self.ent_dim, self.args.attr_dim, self.args.attr_dim, self.args.txt_dim, self.args.name_dim]
        hidden_list = [self.args.vis_dim]

        self.vae = VAE(sum(self.dim[:modal_num]), hidden_list, self.args.vis_sim)

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


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, feat_in, feat_out, attn_dropout, diag=True, init=None, bias=False):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU()
        self.special_spmm = SpecialSpmm()


class SpecialSpmm(nn.Module):
    def forward(self):
        return SpecialSpmm()


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_forward(a, b)
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
        return SpecialSpmmFunction.apply(indices, values, shape, b)


if __name__ == '__main__':
    # 创建测试数据
    indices = torch.tensor([[0, 1, 1], [2, 0, 2]])  # 稀疏矩阵的索引
    values = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)  # 稀疏矩阵的值
    shape = (3, 3)  # 稀疏矩阵的形状
    b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)  # 乘法右侧的稠密矩阵

    # 实例化 SpecialSpmm 并计算前向传播
    spmm = SpecialSpmm()
    output = spmm(indices, values, shape, b)

    # 计算反向传播
    loss = output.sum()  # 构造损失函数
    loss.backward()

    # 打印结果
    print("Output:\n", output)
    print("Grad of values:", values.grad)
    print("Grad of b:\n", b.grad)
