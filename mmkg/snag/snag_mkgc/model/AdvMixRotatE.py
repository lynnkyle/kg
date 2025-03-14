import math
import torch
from torch import nn
from torch import functional as F
from transformers.pytorch_utils import apply_chunking_to_forward


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
        self.ent_str_dim = dim * 2
        self.rel_str_dim = dim
        self.ent_emb = nn.Embedding(num_ent, self.ent_dim)
        self.rel_emb = nn.Embedding(num_rel, self.rel_dim)
        self.vis_emb = nn.Embedding.from_pretrained(vis_emb).requires_grad_(False)
        self.txt_emb = nn.Embedding.from_pretrained(txt_emb).requires_grad_(False)
        self.vis_dim = vis_emb.shape[1]
        self.txt_dim = txt_emb.shape[1]
        self.vis_proj_1 = nn.Linear(self.vis_dim, self.ent_str_dim)
        self.txt_proj_1 = nn.Linear(self.txt_dim, self.ent_str_dim)
        self.vis_proj_2 = nn.Linear(self.ent_str_dim, self.ent_str_dim)
        self.txt_proj_2 = nn.Linear(self.ent_str_dim, self.ent_str_dim)
        """
            四种特征融合的方式
        """
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.ent_attn = nn.Linear(self.ent_str_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        self.learnable_weight = nn.Parameter(torch.ones(3, requires_grad=True))
        self.comb_proj = nn.Linear(self.ent_str_dim * 3, self.ent_str_dim)
        """
            只适用于DB15K数据集
        """
        self.vis_mean = torch.mean(vis_emb, dim=0)
        self.vis_std = torch.std(vis_emb, dim=0)
        self.txt_mean = torch.mean(txt_emb, dim=0)
        self.txt_std = torch.std(txt_emb, dim=0)

    def forward(self, triples):
        """
            noise_update: epoch - 噪声在每个epoch阶段进行更新 batch - 噪声在每个batch阶段进行更新
        """
        h = triples['batch_h']
        t = triples['batch_t']
        r = triples['batch_r']
        mode = triples['mode']  # ??? mode
        h_emb = self.ent_emb(h)
        t_emb = self.ent_emb(t)
        r_emb = self.rel_emb(r)
        if self.args.add_noise == 1 and self.img_proj.training:
            if self.args.noise_update == 'epoch':
                h_txt_emb = self.txt_emb_noise(h)
                t_txt_emb = self.txt_emb_noise(t)
                h_vis_emb = self.vis_emb_noise(h)
                t_vis_emb = self.vis_emb_noise(t)
                h_emb = self.update_ent_noise(h_emb, h)
                t_emb = self.update_ent_noise(t_emb, r)
            else:
                h_txt_emb = self.txt_emb(h)
                h_txt_emb = self.add_noise_to_embed(h_txt_emb, self.txt_mean, self.txt_std, self.args.noise_ratio)
                t_txt_emb = self.txt_emb(t)
                t_txt_emb = self.add_noise_to_embed(t_txt_emb, self.txt_mean, self.txt_std, self.args.noise_ratio)
                h_vis_emb = self.vis_emb(h)
                h_vis_emb = self.add_noise_to_embed(h_vis_emb, self.vis_mean, self.vis_std, self.args.noise_ratio)
                t_vis_emb = self.vis_emb(t)
                t_vis_emb = self.add_noise_to_embed(t_vis_emb, self.vis_mean, self.vis_std, self.args.noise_ratio)
                self.ent_mean = torch.mean(self.ent_emb.weight.data, dim=0)
                self.ent_std = torch.std(self.ent_emb.weight.data, dim=0)
                h_emb = self.add_noise_to_embed(h, self.ent_mean, self.ent_std, self.args.noise_ratio)
                t_emb = self.add_noise_to_embed(t, self.ent_mean, self.ent_std, self.args.noise_ratio)
        else:
            h_vis_emb = self.vis_emb(h)
            t_vis_emb = self.vis_emb(t)
            h_txt_emb = self.txt_emb(h)
            t_txt_emb = self.txt_emb(t)

        if self.args.num_proj == 2:
            h_vis_emb = self.vis_proj_2(self.vis_proj_1(h_vis_emb))
            t_vis_emb = self.vis_proj_2(self.vis_proj_1(t_vis_emb))
            h_txt_emb = self.txt_proj_2(self.txt_proj_1(h_txt_emb))
            t_txt_emb = self.txt_proj_2(self.txt_proj_1(t_txt_emb))
        else:
            h_vis_emb = self.vis_proj_1(h_vis_emb)
            t_vis_emb = self.vis_proj_1(t_vis_emb)
            h_txt_emb = self.txt_proj_1(h_txt_emb)
            t_txt_emb = self.txt_proj_1(t_txt_emb)

        h_joint = self.get_joint_embeddings(h_emb, h_vis_emb, h_txt_emb)
        t_joint = self.get_joint_embeddings(t_emb, t_vis_emb, t_txt_emb)
        score = self.margin - self._calc(h_joint, t_joint, r_emb, mode)
        return score

    """
        高斯噪声适用于epoch阶段
    """

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

    def update_ent_noise(self, ent_emb, batch):
        noise_mask = self.ent_moise_mask[batch]
        selected_emb = ent_emb[noise_mask]
        ent_emb[noise_mask] = (1 - self.args.mask_ratio) * selected_emb + self.args.mask_ratio * self.ent_noise[batch][
            noise_mask]
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

    def get_joint_embeddings(self, str_emb, vis_emb, txt_emb):
        """
        :param str_emb: [batch_size, hidden_size]
        :param vis_emb: [batch_size, hidden_size]
        :param txt_emb: [batch_size, hidden_size]
        :return:
        """
        emb = torch.stack((str_emb, vis_emb, txt_emb), dim=1)  # [batch_size, seq_len, hidden_size]
        u = torch.tanh(emb)  # [batch_size, seq_len, hidden_size]
        hidden_state = u  # [batch_size, seq_len, hidden_size]

        """
            三种特征融合方式: Mformer(mean、 graph、 weight)、 atten_weight、 learnable_weight
        """
        if 'Mformer' in self.args.joint_way:
            for i, layer_module in enumerate(self.fusion_layers):
                layer_output = layer_module(hidden_state, output_attention=True)
                hidden_state = layer_output[0]  # [batch_size, seq_len, hidden_size]
            if 'mean' in self.args.joint_way:
                context_vector = torch.mean(hidden_state, dim=1)  # [batch_size, hidden_size]
            elif 'graph' in self.args.joint_way:
                context_vector = hidden_state[:, 0, :]  # [batch_size, hidden_size]
            elif 'weight' in self.args.joint_way:
                # layer_output[1]: [batch_size, num_attention_head, seq_len, seq_len]
                attention_pro = torch.sum(layer_output[1], dim=-3)
                # [batch_size, seq_len, seq_len]
                attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(3 * self.args.num_attention_head)
                # [batch_size, seq_len]
                attention_weight = F.softmax(attention_pro_comb, dim=-1)
                # [batch_size, seq_len]
                context_vector = torch.sum(attention_weight.unsqueeze(-1) * emb, dim=1)
                # [!!!important]  [batch_size, hidden_size]
            else:
                raise NotImplementedError
        elif 'atten_weight' in self.args.joint_way:
            attention_pro = self.ent_attn(u).squeeze(-1)  # [batch_size, seq_len, 1] -> [batch_size, seq_len]
            attention_weight = torch.softmax(attention_pro, dim=-1)  # [batch_size, seq_len]
            context_vector = torch.sum(attention_weight.unsqueeze(-1) * emb, dim=1)
            # [!!!important]  [batch_size, hidden_size]
        elif 'learnable_weight' in self.args.joint_way:
            learn_weight = torch.softmax(self.learnable_weight, dim=0)  # [seq_len]
            context_vector = torch.sum(learn_weight.unsqueeze(0).unsqueeze(-1) * emb, dim=1)
            # [batch_size, hidden_size]
        else:
            context_vector = self.comb_proj(torch.cat((str_emb, vis_emb, txt_emb), dim=1))  # [batch_size, hidden_size]
        return context_vector

    def _calc(self, h_joint, t_joint, r, mode):
        """
        :param h_joint: [batch_size, seq_len, hidden_size]
        :param t_joint: [batch_size, seq_len, hidden_size]
        :param r: []
        :param mode: head_batch
        :return:
        """
        pi = 3.14159265358979323846
        re_h, im_h = torch.chunk(h_joint, 2, dim=-1)
        re_t, im_t = torch.chunk(t_joint, 2, dim=-1)
        max_r = torch.max(torch.abs(r), dim=1)
        r = r / (max_r / pi)
        re_r, im_r = torch.cos(r), torch.sin(r)
        re_h = re_h.view(re_h.shape[0], -1, re_h.shape[-1])
        # [batch_size, 1, hidden_size]
        im_h = im_h.view(im_h.shape[0], -1, im_h.shape[-1])
        # [batch_size, 1, hidden_size]
        re_t = re_t.view(re_t.shape[0], -1, re_t.shape[-1])
        # [batch_size, 1, hidden_size]
        im_t = im_t.view(im_t.shape[0], -1, im_t.shape[-1])
        # [batch_size, 1, hidden_size]
        re_r = re_r.view(re_r.shape[0], -1, im_r.shape[-1])
        # [batch_size, 1, hidden_size]
        im_r = im_r.view(im_r.shape[0], -1, im_r.shape[-1])
        # [batch_size, 1, hidden_size]
        if mode == 'head_batch':
            re = re_t * re_r - im_t * im_r - re_h  # [batch_size, 1, hidden_size]
            im = re_t * im_r + im_t * re_r - im_h  # [batch_size, 1, hidden_size]
        else:
            re = re_h * re_r - im_h * im_r - re_t  # [batch_size, 1, hidden_size]
            im = re_h * im_r + im_h * re_r - im_t  # [batch_size, 1, hidden_size]
        score = torch.stack([re, im], dim=0)  # [2, batch_size, 1, hidden_size]
        score = torch.sum(torch.norm(score, dim=0), dim=-1)  # [batch_size, 1, hidden_size] -> [batch_size, 1]
        return score.permute(1, 0).flatten()


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_head == 0
        self.num_attention_head = config.num_attention_head
        self.attention_head_size = int(config.hidden_size / config.num_attention_head)
        self.all_head_size = self.num_attention_head * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state, output_attention=False):
        """
        :param hidden_state:    [batch_size, seq_len, hidden_size]
        :param output_attention:
        :return:
        """
        query_layer = self.transpose_for_scores(self.query(hidden_state))
        # [batch_size, num_attention_head, seq_len, attention_head_size]
        key_layer = self.transpose_for_scores(self.key(hidden_state))
        # [batch_size, num_attention_head, seq_len, attention_head_size]
        value_layer = self.transpose_for_scores(self.value(hidden_state))
        # [batch_size, num_attention_head, seq_len, attention_head_size]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [batch_size, num_attention_head, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # [batch_size, num_attention_head, seq_len, seq_len]
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # [batch_size, num_attention_head, seq_len, seq_len]
        attention_probs = self.dropout(attention_probs)
        # [batch_size, num_attention_head, seq_len, seq_len]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [batch_size, num_attention_head, seq_len, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # [batch_size, seq_len, num_attention_head, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # [batch_size, seq_len, all_head_size]
        output = (context_layer, attention_scores) if output_attention else (context_layer,)
        return output

    def transpose_for_scores(self, x):
        """
        :param x:   [batch_size, seq_len, hidden_size]
        :return:    [batch_size, num_attention_head, seq_len, attention_head_size]
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_head, self.attention_head_size)  # [batch_size, seq_len, hidden_size]
        x = x.view(new_x_shape)  # [batch_size, seq_len, num_attention_head, attention_head_size]
        return x.permute(0, 2, 1, 3)  # [batch_size, num_attention_head, seq_len, attention_head_size]


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state, input_tensor):
        """
        :param hidden_state: [batch_size, seq_len, hidden_size]
        :param input_tensor:
        :return:
        """
        hidden_state = self.dense(hidden_state)  # [batch_size, seq_len, hidden_size]
        hidden_state = self.dropout(hidden_state)  # [batch_size, seq_len, hidden_size]
        hidden_state = self.layer_norm(hidden_state + input_tensor)  # [batch_size, seq_len, hidden_size]
        return hidden_state


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = BertSelfAttention(config)
        self.self_out = BertSelfOutput(config)

    def forward(self, hidden_state, output_attention=False):
        self_attn_output = self.self_attn(hidden_state, output_attention)
        # ([batch_size, seq_len, hidden_size], [batch_size, num_attention_head, seq_len, seq_len])
        self_out_output = self.self_out(self_attn_output[0], hidden_state)
        output = (self_out_output,) + self_attn_output[1:]
        # ([batch_size, seq_len, hidden_size], [batch_size, num_attention_head, seq_len, seq_len])
        # 保持 BertSelfAttention 的额外输出, 防止信息丢失
        return output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_state):
        hidden_state = self.dense(hidden_state)  # [batch_size, seq_len, intermediate_size]
        hidden_state = self.intermediate_act_fn(hidden_state)  # [batch_size, seq_len, intermediate_size]
        return hidden_state


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state, input_tensor):
        hidden_state = self.dense(hidden_state)  # [batch_size, seq_len, hidden_size]
        hidden_state = self.dropout(hidden_state)  # [batch_size, seq_len, hidden_size]
        hidden_state = self.layer_norm(hidden_state + input_tensor)  # [batch_size, seq_len, hidden_size]
        return hidden_state


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.config = config
        self.chunk_size_feed_forward = 0  # 划分 seq_len[token数量]
        self.seq_len_dim = 1
        self.attn = BertAttention(config)
        if self.config.use_intermediate:
            self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_state, output_attention=False):
        self_attn_output = self.attn(hidden_state, output_attention)
        if not self.config.use_intermediate:
            return (self_attn_output[0], self_attn_output[1])
        attn_output = self_attn_output[0]  # [batch_size, seq_len, hidden_size]
        output = self_attn_output[1]
        """
            apply_chunking_to_forward 的主要作用是将输入序列分块，减少每次前向传播时的内存占用，特别是当输入序列非常长时
        """
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        output = (layer_output, output)
        return output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # ([batch_size, seq_len, hidden_size], [batch_size, num_attention_head, seq_len, seq_len])
        return layer_output
