import torch
from dgl import DGLGraph
from torch import nn


class MMKGModel(nn.Module):
    def __init__(self, num_ent, num_rel, h_dim):
        super().__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.entity_emb = nn.Parameter(torch.tensor((num_ent, h_dim), dtype=torch.float), requires_grad=True)
        self.rel_emb = nn.Parameter(torch.tensor((num_rel * 2, h_dim), dtype=torch.float), requires_grad=True)
        self.encoder = None
        self.decoder = None
        self.loss_e = nn.CrossEntropyLoss()
        self.loss_r = nn.CrossEntropyLoss()

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
        # self.eval()
        with torch.no_grad():
            inverse_triplets = origin_triplets[:, [2, 1, 0]]
            inverse_triplets[:, 1] += self.num_rel
            triplets = torch.cat((origin_triplets, inverse_triplets))
            score_ent, score_rel = self.forward(triplet_list=triplets)
            return triplets, score_ent, score_rel

    """
        get_metrics: mrr, filter_mrr, rank, filter_rank
    """

    def get_metrics(self, test_triplets, eval_batch_size=1000, hits=(1, 3, 10)):
        raw_rank = []
        filter_rank = []
        num_triples = len(test_triplets)
        self.eval()
        with torch.no_grad():
            test_triplets, score_ent, score_rel = self.predict(test_triplets)
            n_batch = (num_triples + eval_batch_size - 1) // eval_batch_size  # [!!!important]
            for idx in range(n_batch):
                batch_start = idx * eval_batch_size
                batch_end = min((idx + 1) * eval_batch_size, num_triples)
                triple_batch = test_triplets[batch_start:batch_end, :]
                score_ent_batch = score_ent[batch_start:batch_end, :]
                score_rel_batch = score_rel[batch_start:batch_end, :]
                target_ent = test_triplets[batch_start:batch_end, 2]  # 目标实体
                target_rel = test_triplets[batch_start:batch_end, 1]  # 目标关系
                raw_rank.append(sort_and_rank())
                filter_rank.append(sort_and_rank())

    """
        sort_and_rank:
        score: [batch_size, num_ent]
        target: [batch_size, 1]
        Eg:
            score = torch.randn(size=(15, 6))
            print(score)
            target = torch.randint(size=(15, 1), low=1, high=6)
            print(target)
            model = MMKGModel(num_ent=6, num_rel=3, h_dim=4)
            print(model.sort_and_rank(score, target))
    """

    def sort_and_rank(self, score, target):
        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1))
        indices = indices[:, 1].view(-1)
        return indices

    """
        sort_and_rank_filter:
        score: [batch_size, num_ent]
        target: [batch_size, 1]
        Eg:
            
    """

    def sort_and_rank_filter(self, batch_head, batch_rel, score, target, all_ans):
        for i in range(len(batch_head)):
            # 过滤除目标实体的所有三元组
            ans = target[i]
            multi = list(all_ans[batch_head[i].item()][batch_rel[i].item()])
            ground = score[i][ans]
            score[i][multi] = 0
            score[i][ans] = ground
        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1))
        indices = indices[:, 1].view(-1)
        return indices


if __name__ == '__main__':
    pass
