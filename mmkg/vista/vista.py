import torch
from torch import nn


class VISTA(nn.Module):
    def __init__(self, num_ent, num_rel, dim_str, ent_vis, rel_vis, dim_vis, ent_txt, rel_txt, dim_txt, ent_vis_mask,
                 rel_vis_mask, num_head, dim_ffn, num_layer_enc_ent, num_layer_enc_rel, num_layer_dec,
                 dropout=0.1, str_dropout=0.6, vis_dropout=0.1, txt_dropout=0.1):
        super(VISTA, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.dim_str = dim_str

        self.ent_vis = ent_vis
        self.rel_vis = rel_vis
        self.ent_txt = ent_txt.unsqueeze(dim=1)
        self.rel_txt = rel_txt.unsqueeze(dim=1)

        false_ents = torch.full((self.num_ent, 1), False).cuda()
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, false_ents], dim=1)
        false_rels = torch.full((self.num_rel, 1), False).cuda()
        self.rel_mask = torch.cat([false_rels, false_rels, rel_vis_mask, false_rels], dim=1)

        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_emb = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_emb = nn.Parameter(torch.Tensor(num_rel, 1, dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1, dim_str))  # 解码器部分的[Mask] 拼接在ent_emb的最后一行

        self.str_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.str_dr = nn.Dropout(p=str_dropout)
        self.vis_dr = nn.Dropout(p=vis_dropout)
        self.txt_dr = nn.Dropout(p=txt_dropout)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.proj_ent_vis = nn.Linear(dim_vis, dim_str)
        self.proj_rel_vis = nn.Linear(dim_vis * 3, dim_str)
        self.proj_txt = nn.Linear(dim_txt, dim_str)

        self.num_head = num_head
        self.dim_hid = dim_ffn
        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_ffn, dropout, batch_first=True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_ffn, dropout, batch_first=True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_ffn, dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.ent_emb)
        nn.init.xavier_uniform_(self.rel_emb)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_rel_vis.weight)
        nn.init.xavier_uniform_(self.proj_txt.weight)

        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

        self.proj_ent_vis.bias.data.zero_()
        self.proj_rel_vis.bias.data.zero_()
        self.proj_txt.bias.data.zero_()

    def forward(self):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.str_dr(self.str_ln(self.ent_emb)) + self.pos_str_ent
        rep_ent_vis = self.vis_dr(self.vis_ln(self.proj_ent_vis(self.ent_vis))) + self.pos_vis_ent
        rep_ent_txt = self.txt_dr(self.txt_ln(self.proj_txt(self.ent_txt))) + self.pos_txt_ent
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)[:, 0]

        rel_tkn = self.rel_token.tile(self.num_rel, 1, 1)
        rep_rel_str = self.str_dr(self.str_ln(self.rel_emb)) + self.pos_str_rel
        rep_rel_vis = self.vis_dr(self.vis_ln(self.proj_rel_vis(self.rel_vis))) + self.pos_vis_rel
        rep_rel_txt = self.txt_dr(self.txt_ln(self.proj_txt(self.rel_txt))) + self.pos_txt_rel
        rel_seq = torch.cat([rel_tkn, rep_rel_str, rep_rel_vis, rep_rel_txt], dim=1)
        rel_embs = self.rel_encoder(rel_seq, src_key_padding_mask=self.rel_mask)[:, 0]
        return torch.cat([ent_embs, self.lp_token], dim=0), rel_embs  # ent_embs的最后一行是[可学习的Mask]

    def score(self, ent_embs, rel_embs, triples):
        """
        :param ent_embs:    [num_entity, emb_dim]
        :param rel_embs:    [num_entity, emb_dim]
        :param triples:     [batch_size, 3] 二维张量
        :return:
        """
        h_seq = ent_embs[triples[:, 0] - self.num_rel].unsqueeze(1) + self.pos_head  # [batch_size, 1, emb_dim]
        r_seq = rel_embs[triples[:, 1] - self.num_ent].unsqueeze(1) + self.pos_rel  # [batch_size, 1, emb_dim]
        t_seq = ent_embs[triples[:, 2] - self.num_rel].unsqueeze(1) + self.pos_tail  # [batch_size, 1, emb_dim]
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)  # [batch_size, 3, emb_dim]
        output_dec = self.decoder(dec_seq)[triples == self.num_ent + self.num_rel]  # [batch_size, 1, emb_dim]
        score = torch.inner(output_dec, ent_embs)  # [batch_size, num_entity]
        return score
