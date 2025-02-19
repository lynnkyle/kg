import torch
from dgl import DGLGraph
from torch import nn


class MMKGModel(nn.Module):
    def __init__(self, num_entity, num_rel, h_dim, entity_prediction, relation_prediction):
        super().__init__()
        self.num_entity = num_entity
        self.num_rel = num_rel
        self.entity_emb = nn.Parameter(torch.tensor((num_entity, h_dim), dtype=torch.float), requires_grad=True)
        self.rel_emb = nn.Parameter(torch.tensor((num_rel * 2, h_dim), dtype=torch.float), requires_grad=True)
        self.encoder = None
        self.decoder = None
        self.loss_e = nn.CrossEntropyLoss()
        # self.loss_r = nn.CrossEntropyLoss()

    #   排序后的(预测实体, 预测实体得分)
    def forward(self, g_list: DGLGraph = None, triplet_list=None):
        pass

    """
        get_loss:
            origin_triplets: 训练集
    """

    def get_loss(self, origin_triplets):
        loss_ent = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        loss_rel = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        inverse_triplets = origin_triplets[:, [2, 1, 0]]
        inverse_triplets[:, 1] += self.num_rel
        triplets = torch.cat((origin_triplets, inverse_triplets))
        score_ent = self.forward(triplet_list=triplets)
        score_rel = self.forward(triplet_list=triplets)
        if self.entity_prediction:
            loss_ent += self.loss_e(score_ent, triplets[:, 2])
        if self.relation_prediction:
            loss_rel += self.loss_r(score_rel, triplets[:, 1])
        # self.loss_r()
        return

    """
        predict:
            origin_triplets: 测试集
    """

    def predict(self, origin_triplets=None):
        # model.eval()
        with torch.no_grad():
            inverse_triplets = origin_triplets[:, [2, 1, 0]]
            inverse_triplets[:, 1] += self.num_rel
            triplets = torch.cat((origin_triplets, inverse_triplets))
            score_ent, score_rel = self.forward(triplet_list=triplets)
            return triplets, score_ent, score_rel

    """
        get_metrics: mrr, filter_mrr, rank, filter_rank
    """

    def get_metrics(self, test_triplets, hits=(1, 3, 10)):
        rank_at = torch.cat(hits)
        total_rank = torch.cat(rank_list)
        model.eval()
        with torch.no_grad():



if __name__ == '__main__':
    input = torch.tensor([[1, 2, 1], [2, 3, 1], [3, 1, 2]])
    model = MMKGModel(num_entity=3, num_rel=3, h_dim=2)
    print(model.get_loss(input))
