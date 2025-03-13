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
        self.args = args
        self.margin = margin
        self.epsilon = epsilon
        self.ent_dim = dim
        self.rel_dim = dim * 2
        self.ent_emb = nn.Embedding(num_ent, self.ent_dim)
        self.rel_emb = nn.Embedding(num_rel, self.rel_dim)
        self.vis_emb = nn.Embedding.from_pretrained(vis_emb).requires_grad_(False)
        self.txt_emb = nn.Embedding.from_pretrained(txt_emb).requires_grad_(False)
        """
            只适用于DB15K数据集
        """
        self.vis_mean = torch.mean(vis_emb, dim=0)
        self.vis_std = torch.std(vis_emb, dim=0)
        self.txt_mean = torch.mean(txt_emb, dim=0)
        self.txt_std = torch.std(txt_emb, dim=0)

    def forward(self, triples):
        h = triples['batch_h']
        t = triples['batch_t']
        r = triples['batch_r']
        mode = triples['mode']  # ??? mode
        h_emb = self.ent_emb(h)
        t_emb = self.ent_emb(t)
        r_emb = self.rel_emb(r)
        if self.args.add_noise == 1 and self.img_proj.training:
            if self.args.noise_update == 'epoch':
                h_txt = self.txt_emb_noise(h)
                t_txt = self.txt_emb_noise(t)
                h_vis = self.vis_emb_noise(h)
                t_vis = self.vis_emb_noise(t)
                h_emb_noise = self.update_ent_noise(h_emb, h)
                t_emb_noise = self.update_ent_noise(t_emb, r)
            else:
                pass

    def update_noise(self):
        txt_noise_weights = self.add_noise_to_embed(self.txt_emb.weight.data.clone(), self.txt_mean, self.txt_std,
                                                    self.args.noise_ratio)
        self.txt_emb_noise = torch.nn.Embedding.from_pretrained(txt_noise_weights).requires_grad_(False)

        vis_noise_weights = self.add_noise_to_embed(self.vis_emb.weight.data.clone(), self.vis_mean, self.vis_std,
                                                    self.args.noise_ratio)
        self.vis_emb_noise = torch.nn.Embedding.from_pretrained(vis_noise_weights).requires_grad_(False)

        self.ent_mean = torch.mean(self.ent_emb.weight.data, dim=0)
        self.ent_std = torch.std(self.rel_emb.weight.data, dim=0)
        self.ent_noise = self.ent_mean + self.ent_std * torch.randn_like(self.ent_emb.weight.data)
        self.ent_moise_mask = torch.rand(self.ent_emb.weight.shape[0]) < self.args.noise_ratio

    """
    """

    def update_ent_noise(self, ent_emb, batch):
        noise_mask = self.ent_moise_mask[batch]
        selected_emb = ent_emb[noise_mask]
        ent_emb[noise_mask] = (1 - self.args.mask_ratio) * selected_emb + self.args.mask_ratio * self.ent_noise[batch][noise_mask]
        return ent_emb

    """
         GMNM: Gauss Modality Noise Masking; 图像和文本添加高斯噪声
         相较于Dropout和Bert-style Masking, GMNM不只是丢弃信息, 而是用高斯噪声扰动数据, 让模型学会在不确定的环境下仍然能有效学习特征
    """

    def add_noise_to_embed(self, emb, mean, std, noise_ratio=0.1):
        """
        :param emb:  [12842, 768]
        :param mean:    每幅图像每个维度的平均值
        :param std:     每幅图像每个维度的标准差
        :param noise_ratio: 超参数
        :return:
        """
        noise_mask = torch.rand(emb.shape[0]) < noise_ratio
        if not noise_mask.any():
            return emb
        selected_emb = emb[noise_mask]
        noise = mean + std * torch.randn_like(selected_emb)  # 高斯噪声满足 N(mean, std)
        emb[noise_mask] = (1 - self.args.mask_ratio) * selected_emb + self.args.mask_ratio * noise
        return emb
