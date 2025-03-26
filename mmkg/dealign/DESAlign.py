import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import apply_chunking_to_forward

from mmkg.dealign.DESAlign_tools import VirEmbGen_vae


class DESAlign(nn.Module):
    def __init__(self, kgs, args):
        super(DESAlign, self).__init__()

    def forward(self, batch):
        pass


class MultiModalEncoder(nn.Module):
    def __init__(self, args, ent_num, vis_feat_dim=None, txt_feat_dim=None, use_project_head=False,
                 rel_input_dim=1000, attr_input_dim=1000, name_input_dim=300):
        super().__init__()
        self.args = args
        self.ent_num = ent_num
        attr_dim = self.args.attr_dim
        name_dim = self.args.name_dim
        vis_dim = self.args.vis_dim
        txt_dim = self.args.txt_dim
        dropout = self.args.dropout
        self.use_project_head = use_project_head
        """ 隐藏层维度列表 """
        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])
        """ GAT注意力头列表 """
        self.n_gat_heads = [int(x) for x in self.args.gat_heads.strip().split(",")]
        """
            实体嵌入层
        """
        self.ent_emb = nn.Embedding(self.ent_num, self.input_dim)
        nn.init.normal_(self.ent_emb.weight, std=1 / math.sqrt(self.ent_num))
        # nn.init.normal_(self.ent_emb.weight, std=1 / math.sqrt(self.input_dim))
        self.ent_emb.requires_grad = True
        """
            模态编码器
        """
        self.rel_fc = nn.Linear(rel_input_dim, attr_dim)
        self.attr_fc = nn.Linear(attr_input_dim, attr_dim)
        self.name_fc = nn.Linear(name_input_dim, name_dim)
        self.vis_fc = nn.Linear(vis_feat_dim, vis_dim)
        self.txt_fc = nn.Linear(txt_feat_dim, txt_dim)

        self.vir_emb_gen_vae = VirEmbGen_vae(args)  # ???

        self.fusion = MformerFusion(args)

    def forward(self, input_idx, adj, rel_feat=None, attr_feat=None, name_feat=None, vis_feat=None, txt_feat=None,
                ent_wo_vis=None, _test=False):


class MformerFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight_raw = torch.nn.Parameter(torch.ones(6, requires_grad=True))
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])

    def forward(self, embs):
        """
        :param embs:    list: 6 [num_modal, emb_dim]
        :return:    []
        """
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]  # list: n [num_ent, emb_dim]
        num_modal = len(embs)

        hidden_states = torch.stack(embs, dim=1)  # [num_ent, num_modal, emb_dim]
        layer_output = None
        for i, layer in enumerate(self.fusion_layer):
            layer_output = layer(hidden_states, output_attentions=True)
            hidden_states = layer_output[0]
        attention_pro = torch.sum(layer_output[1], dim=-3)
        # [num_ent, num_attention_head, seq_len, seq_len] -> [num_ent, seq_len, seq_len] (query, key)
        attention_pro_comb = torch.sum(attention_pro, dim=-1) / math.sqrt(num_modal * self.args.num_attention_heads)
        # [num_ent, seq_len]
        weight_norm = F.softmax(attention_pro_comb, dim=-1)
        # [num_ent, seq_len]
        embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(num_modal)]
        # [num_ent, 1] [num_ent, emb_dim]   ->  list: 4 [num_ent, emb_dim]
        joint_embs = torch.cat(embs, dim=1)  # [num_ent, 4 * emb_dim]
        weight_norm_fz = F.softmax(self.weight_raw, dim=0)
        # [6, ]
        # [!!!important]   weight_norm_fz = F.softmax(self.weight_raw[:num_modal], dim=0)
        embs_fz = [weight_norm_fz[idx] * F.normalize(embs[idx]) for idx in range(num_modal)]
        joint_embs_fz = torch.cat(embs_fz, dim=1)
        return joint_embs, joint_embs_fz, hidden_states, weight_norm


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
        output = (context_layer, attention_probs) if output_attention else (context_layer,)
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
