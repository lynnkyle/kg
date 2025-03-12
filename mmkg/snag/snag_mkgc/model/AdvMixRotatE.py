import torch
from torch import nn
import  torchvision.transforms as transforms

transforms.GaussianBlur


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

    def forward(self, triples):
        h = triples['batch_h']
        t = triples['batch_t']
        r = triples['batch_r']
        mode = triples['mode']  # ??? mode
        h_emb = self.ent_emb(h)
        t_emb = self.ent_emb(t)
        r_emb = self.rel_emb(r)
        if self.args.add_noise == 1 and self.img_proj.training:

    def add_noise_to_embed(self, embeddings, mean, std, noise_ratio=0.1):
        noise_mask = torch.rand(embeddings.shape[0]) < noise_ratio
        if not noise_mask.any():
            return embeddings
        selected_embeddings = embeddings[noise_mask]
        noise = mean + std * torch.randn_like()
