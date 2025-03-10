import torch
from torch import nn


class AdvMixRotatE(nn.Module):
    def __init__(self, args, num_ent, num_rel, dim=100, margin=6.0, epsilon=2.0, vis_emb=None, txt_emb=None):
        super(AdvMixRotatE, self).__init__()
        assert vis_emb is not None
        assert txt_emb is not None
        """
            RotatE初始化的参数
        """
        self.margin = margin
        self.epsilon = epsilon
