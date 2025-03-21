import torch
from torch import nn


class DESAlign(nn.Module):
    def __init__(self, kgs, args):
        super(DESAlign, self).__init__()

    def forward(self, batch):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, joint_emb_fz, hidden_states, weight_norm, vir_emb, x_hat,
        hyb_emb, kl_div = self.multi
