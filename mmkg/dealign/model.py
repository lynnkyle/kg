from typing import Any

import math
import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, emb_size, hid_size_list, mid_hid):
        super(VAE, self).__init__()
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
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
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
    def __init__(self, in_features, out_features):
        super().__init__()
        pass

    def forward(self, x, adj):
        pass


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shapes, b):
        pass


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shapes, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values)
        ctx.save_for_backward(a, b)
        ctx.N = shapes[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a = torch.matmul(grad_output, b.t())
            grad_values = torch.mm()

        if ctx.needs_input_grad[3]:
            grad_b = torch.matmul(grad_output, a.t())

        return grad_values, grad_b, None, None
