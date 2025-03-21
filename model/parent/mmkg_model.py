import torch
from dgl import DGLGraph
from torch import nn


class MMKGModel(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super().__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.ent_emb = nn.Parameter(torch.Tensor(size=(num_ent, emb_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.ent_emb)
        self.rel_emb = nn.Parameter(torch.Tensor(size=(num_rel * 2, emb_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.rel_emb)
        self.encoder = None
        self.decoder = None
        self.loss_e = nn.CrossEntropyLoss()
        self.loss_r = nn.CrossEntropyLoss()

    #   排序后的(预测实体, 预测实体得分)
    def forward(self, triple_list=None, g_list: DGLGraph = None):
        x = self.decoder.forward(self.ent_emb, self.rel_emb, triple_list)  # [!!!important] [batch_size, num_entity]
        return x

    """
        get_loss:
            origin_triplets: 训练集
    """

    def get_loss(self, origin_triplets):
        loss_ent = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        # loss_rel = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        inverse_triplets = origin_triplets[:, [2, 1, 0]]
        inverse_triplets[:, 1] += self.num_rel
        triplets = torch.cat((torch.from_numpy(origin_triplets), torch.from_numpy(inverse_triplets)))
        triplets = triplets.to(self.device)
        score_ent = self.forward(triple_list=triplets)
        # if self.entity_prediction:
        loss_ent += self.loss_e(score_ent, triplets[:, 2])
        # if self.relation_prediction:
        #     loss_rel += self.loss_r(score_rel, triplets[:, 1])
        # self.loss_r()
        return loss_ent

    """
        predict:
            origin_triplets: 测试集
    """

    def predict(self, origin_triplets=None):
        # self.eval()
        with torch.no_grad():
            inverse_triplets = origin_triplets[:, [2, 1, 0]]
            inverse_triplets[:, 1] += self.num_rel
            triplets = torch.cat((torch.from_numpy(origin_triplets), torch.from_numpy(inverse_triplets)))
            score_ent = self.forward(triple_list=triplets)
            score_rel = []
            return triplets, score_ent, score_rel

    """
        get_metrics: mrr, filter_mrr, rank, filter_rank
    """

    def get_metrics(self, test_triplets, eval_batch_size=1000, hits=(1, 3, 10)):
        raw_ent_rank = []
        filter_ent_rank = []
        raw_rel_rank = []
        filter_rel_rank = []
        to_skip_ent = []
        to_skip_rel = []
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
                # score_rel_batch = score_rel[batch_start:batch_end, :]
                target_ent_batch = test_triplets[batch_start:batch_end, 2]  # 目标实体
                # target_rel_batch = test_triplets[batch_start:batch_end, 1]  # 目标关系
                raw_ent_rank.append(self.sort_and_rank(score_ent_batch, target_ent_batch))
                # filter_ent_rank.append(
                #     self.sort_and_rank_filter(triple_batch, score_ent_batch, target_ent_batch, to_skip_ent, predict=0))
                # raw_rel_rank.append(self.sort_and_rank(score_rel_batch, target_rel_batch))
                # filter_rel_rank.append(
                #     self.sort_and_rank_filter(triple_batch, score_rel_batch, target_rel_batch, to_skip_rel, predict=1))
                ent_raw_mrr, ent_raw_hits = self.get_mrr(raw_ent_rank), self.get_hits(raw_ent_rank)
                # ent_filter_mrr, ent_filter_hits = self.get_mrr(filter_ent_rank), self.get_hits(filter_ent_rank)
                # rel_raw_mrr, rel_raw_hits = self.get_mrr(raw_rel_rank), self.get_hits(raw_rel_rank)
                # rel_filter_mrr, rel_filter_hits = self.get_mrr(filter_rel_rank), self.get_hits(filter_rel_rank)
        return ("Metrics: \n" +
                "Entity:  \n" +
                ("raw:       mrr: {:.4f} Hit@1: {:.4f} Hit@3: {:.4f} Hit@10: {:.4f}\n".format(ent_raw_mrr,
                                                                                              ent_raw_hits[0],
                                                                                              ent_raw_hits[1],
                                                                                              ent_raw_hits[2])))

    # + ("filter:    mrr: {:.4f} Hit@1: {:.4f} Hit@3: {:.4f} Hit@10: {:.4f}\n".format(ent_filter_mrr,
    #                                                                                 ent_filter_hits[0],
    #                                                                                 ent_filter_hits[1],
    #                                                                                 ent_filter_hits[2]))
    # + "Relation:\n" +
    # ("raw:       mrr: {:.4f} Hit@1: {:.4f} Hit@3: {:.4f} Hit@10: {:.4f}\n".format(rel_raw_mrr,
    #                                                                               ent_raw_hits[0],
    #                                                                               ent_raw_hits[1],
    #                                                                               ent_raw_hits[2])) +
    # ("filter:    mrr: {:.4f} Hit@1: {:.4f} Hit@3: {:.4f} Hit@10: {:.4f}\n").format(rel_filter_mrr,
    #                                                                                rel_filter_hits[0],
    #                                                                                rel_filter_hits[1],
    #                                                                                rel_filter_hits[2]))

    """
        sort_and_rank:
        score: [batch_size, num_ent]
        target: [batch_size, 1]
        Eg:
            score = torch.randn(size=(15, 6))
            print(score)
            target = torch.randint(size=(15, 1), low=1, high=6)
            print(target)
            model_0 = MMKGModel(num_ent=6, num_rel=3, h_dim=4)
            print(model_0.sort_and_rank(score, target))
    """

    def sort_and_rank(self, score, target):
        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices.detach().cpu() == target.view(-1, 1))
        indices = indices[:, 1].view(-1)
        return indices

    """
        sort_and_rank_filter:
        score: [batch_size, num_ent]
        target: [batch_size, 1]
        Eg:

    """

    def sort_and_rank_filter(self, triple_batch, score, target, to_skip, predict):
        if predict == 0:  # 实体预测
            first = triple_batch[:, 0]
            second = triple_batch[:, 1]
        else:  # 关系预测
            first = triple_batch[:, 0]
            second = triple_batch[:, 2]
        for i in range(len(triple_batch)):
            # 过滤除目标实体的所有三元组
            ans = target[i]
            multi = list(to_skip[first[i].item()][second[i].item()])
            ground = score[i][ans]
            score[i][multi] = 0
            score[i][ans] = ground
        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1))
        indices = indices[:, 1].view(-1)
        return indices

    """
        get_mrr: mrr评估指标
        Eg:
            for i in range(3):
                rank = torch.randint(low=0, high=6, size=(8,))
                rank_all.append(rank)
            print(rank_all)
            mrr = model_0.get_mrr(rank_all)
            print(mrr)
    """

    def get_mrr(self, rank):
        rank_all = torch.cat(rank)
        rank_all += 1
        mrr = torch.mean(1 / rank_all)
        return mrr

    """
        get_hits: hits评估指标
        Eg:
            for i in range(3):
                rank = torch.randint(low=0, high=6, size=(8,))
                rank_all.append(rank)
            print(rank_all)
            hits = model_0.get_hits(rank_all, hits=(1, 3, 10))
            print(hits)
    """

    def get_hits(self, rank, hits=(1, 3, 10)):
        rank_all = torch.cat(rank)
        rank_all += 1
        hits_res = []
        for hit in hits:
            avg_count = torch.mean((rank_all <= hit).float())
            hits_res.append(avg_count)
        return hits_res
