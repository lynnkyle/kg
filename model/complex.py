import torch
from torch import nn


class ComplEx(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device=None):
        super(ComplEx, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.ent_emb = nn.Parameter(torch.zeros(self.num_ent, self.emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.ent_emb)
        self.rel_emb = nn.Parameter(torch.zeros(self.num_rel * 2, self.emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.rel_emb)
        self.device = device
        self.loss_e = nn.CrossEntropyLoss()

    def forward(self, triples):
        head = self.ent_emb[triples[:, 0]].unsqueeze(1)  # [batch_size, 1, embed_dim]
        rel = self.rel_emb[triples[:, 1]].unsqueeze(1)  # [batch_size, 1, embed_dim]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_rel, im_rel = torch.chunk(rel, 2, dim=2)
        re_ent, im_ent = torch.chunk(self.ent_emb, 2, dim=1)
        re_ans = re_head * re_rel - im_head * im_rel
        im_ans = re_head * im_rel + im_head * re_rel
        score = re_ans * re_ent + im_ans * im_ent
        score = score.sum(dim=2)
        return score

    def get_loss(self, origin_triples):
        loss_ent = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        # loss_rel = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        inverse_triplets = origin_triples[:, [2, 1, 0]]
        inverse_triplets[:, 1] += self.num_rel
        triples = torch.cat((torch.from_numpy(origin_triples), torch.from_numpy(inverse_triplets)))
        triples = triples.to(self.device)
        score_ent = self.forward(triples=triples)
        # if self.entity_prediction:
        loss_ent += self.loss_e(score_ent, triples[:, 2])
        # if self.relation_prediction:
        #     loss_rel += self.loss_r(score_rel, triples[:, 1])
        # self.loss_r()
        return loss_ent


if __name__ == '__main__':
    x = torch.tensor(data=[[0, 1, 2], [2, 2, 1], [1, 2, 0]])
    model = ComplEx(num_ent=3, num_rel=3, emb_dim=6, device='cuda:0')
    print(model(x))
