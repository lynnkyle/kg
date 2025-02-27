import torch
from torch import nn


class RotatE(nn.Module):
    """
        通过复数空间哈达玛积来实现旋转
    """

    def __init__(self, num_ent, num_rel, emb_dim, gamma, device=None):
        super(RotatE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.tensor([gamma]), requires_grad=False
        )
        self.device = device
        self.emb_range = nn.Parameter(
            torch.tensor([(self.gamma.item() + self.epsilon) / emb_dim]), requires_grad=False
        )
        self.ent_dim = emb_dim * 2
        self.rel_dim = emb_dim
        self.ent_emb = nn.Parameter(torch.zeros(self.num_ent, self.ent_dim), requires_grad=True)
        nn.init.uniform_(self.ent_emb, -self.emb_range.item(), self.emb_range.item())
        self.rel_emb = nn.Parameter(torch.zeros(self.num_rel * 2, self.rel_dim), requires_grad=True)
        nn.init.uniform_(self.rel_emb, -self.emb_range.item(), self.emb_range.item())
        self.loss_e = nn.CrossEntropyLoss()

    def forward(self, triples):
        head = self.ent_emb[triples[:, 0], :].unsqueeze(1)
        rel = self.rel_emb[triples[:, 1], :].unsqueeze(1)
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_ent, im_ent = torch.chunk(self.ent_emb, 2, dim=1)
        phase_rel = rel / (self.emb_range.item() / pi)
        re_rel = torch.cos(phase_rel)
        im_rel = torch.sin(phase_rel)
        re_score = re_head * re_rel - im_head * im_rel - re_ent
        im_score = re_head * im_rel + im_head * re_rel - im_ent
        score = torch.stack((re_score, im_score), dim=0)
        score = score.norm(dim=0)
        score = self.gamma - torch.sum(score, dim=-1)
        return score

    def get_loss(self, origin_triplets):
        loss_ent = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        # loss_rel = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        inverse_triplets = origin_triplets[:, [2, 1, 0]]
        inverse_triplets[:, 1] += self.num_rel
        triples = torch.cat((torch.from_numpy(origin_triplets), torch.from_numpy(inverse_triplets)))
        triples = triples.to(self.device)
        score_ent = self.forward(triples=triples)
        # if self.entity_prediction:
        loss_ent += self.loss_e(score_ent, triples[:, 2])
        # if self.relation_prediction:
        #     loss_rel += self.loss_r(score_rel, triples[:, 1])
        # self.loss_r()
        return loss_ent


class pRotatE(nn.Module):
    """
        通过相位差来实现旋转
    """

    def __init__(self, num_ent, num_rel, emb_dim, device=None):
        super(pRotatE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.ent_emb = nn.Parameter(torch.zeros(self.num_ent, self.emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.ent_emb)
        self.rel_emb = nn.Parameter(torch.zeros(self.num_rel * 2, self.emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.rel_emb)
        self.max_ent_value = torch.max(torch.abs(self.ent_emb))
        self.max_rel_value = torch.max(torch.abs(self.rel_emb))
        self.loss_e = nn.CrossEntropyLoss()

    def forward(self, triples):
        head = self.ent_emb[triples[:, 0], :].unsqueeze(1)
        rel = self.rel_emb[triples[:, 1], :].unsqueeze(1)
        pi = 3.14159265358979323846
        phase_head = head / (self.max_ent_value.item() / pi)
        phase_rel = rel / (self.max_rel_value.item() / pi)
        phase_ent = self.ent_emb / (self.max_ent_value.item() / pi)
        score = phase_head + phase_rel - phase_ent
        score = torch.sin(score)
        score = -torch.sum(score, dim=2)
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
    model = RotatE(num_ent=5, num_rel=4, emb_dim=6, gamma=0.9)
    triple = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    model(triple)
