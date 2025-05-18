import json
import os
import random
from copy import deepcopy
from collections import defaultdict
import numpy as np
import networkx as nx
import torch


def load_triples(file_path):
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            triplets.append((h, r, t))
    return triplets


def load_text(data_dir, tokenizer, max_seq_len):
    def truncate_text(ent2text, tokenizer, max_len=50):
        ents, texts = [], []
        for k, v in ent2text.items():
            ents.append(k)
            texts.append(v)

        encoded = tokenizer(
            texts, add_special_tokens=False, padding=True, truncation=True, max_length=max_len, return_tensors='pt',
            return_token_type_ids=False, return_attention_mask=False
        )

        input_ids = encoded['input_ids']
        truncated_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        assert len(ents) == len(truncated_texts)
        return {ent: truncated_texts[idx] for idx, ent in enumerate(ents)}

    ent2text = json.load(open(os.path.join(data_dir, 'entity.json'), 'r', encoding='utf-8'))
    ent2name = {k: ent2text[k]['name'] for k in ent2text}
    ent2desc = {k: ent2text[k]['desc'] for k in ent2text}
    ent2desc = truncate_text(ent2desc, tokenizer, max_seq_len)
    rel2text = json.load(open(os.path.join(data_dir, 'relation.json'), 'r', encoding='utf-8'))
    rel2name = {k: rel2text[k]['name'] for k in rel2text}

    return ent2name, ent2desc, rel2name


class RelationOccurrence(object):
    def __init__(self, data_dir):
        self.triples = load_triples(os.path.join(data_dir, 'train.txt'))
        self.rel = self.get_relations()
        self.one_hop_triples, self.one_hop_relations = self.get_one_hop_triples()
        self.rel_occurrences = self.count_rel_occurrences()

    def get_relations(self):
        rel = set()
        for h, r, t in self.triples:
            rel.add(r)
        return rel

    def get_one_hop_triples(self):
        """
        获取one_hop三元组的信息
        one_hop_triples: {1: {(1, 'likes', 2)}, 2: {(1, 'likes', 2), (2, 'knows', 3)}, 3: {(2, 'knows', 3)}}
        one_hop_relations: {1: {('likes', 0)}, 2: {('likes', 1), ('knows', 0)}, 3: {('knows', 1)}}
        :return:
        """
        one_hop_triples = defaultdict(set)
        one_hop_relations = defaultdict(set)
        for h, r, t in self.triples:
            one_hop_triples[h].add((h, r, t))
            one_hop_triples[t].add((h, r, t))
            one_hop_relations[h].add((r, 0))
            one_hop_relations[t].add((r, 1))
        return one_hop_triples, one_hop_relations

    def count_rel_occurrences(self):
        """
        计算成对关系的共现次数
        :return:
        """
        rel_occurrences = defaultdict(set)
        for entity, one_hop_triple in self.one_hop_triples.items():
            for h, r, t in one_hop_triple:
                for r_sample, direct in self.one_hop_relations[entity]:
                    if r == r_sample:
                        continue
                    elif entity == h:
                        rel_occurrences[((r, 0), (r_sample, direct))] += 1
                    else:
                        rel_occurrences[((r, 1), (r_sample, direct))] += 1
        return rel_occurrences


class KnowledgeGraph(object):
    def __init__(self, args, tokenizer):
        self.args = args
        # Ent、Rel Information
        self.ent2name, self.ent2desc, self.rel2name = load_text(args.data_dir, tokenizer, args.max_seq_len)
        self.id2ent = {idx: ent for idx, ent in enumerate(self.ent2name.keys())}
        self.ent2id = {ent: idx for idx, ent in self.id2ent.items()}

        # Triplets Information
        self.train_triplets = load_triples(os.path.join(args.data_dir, 'train.txt'))
        self.valid_triplets = load_triples(os.path.join(args.data_dir, 'valid.txt'))
        self.test_triplets = load_triples(os.path.join(args.data_dir, 'test.txt'))

        # All Entity AND All Relation
        triplets = self.train_triplets
        self.ent_list = sorted(list(set([h for h, _, _ in triplets] + [t for _, _, t in triplets])))
        self.rel_list = sorted(list(set([r for _, r, _ in triplets])))
        print(f'entity num: {len(self.ent_list)}; relation num: {len(self.rel_list)}')
        self.relation_occurrence = RelationOccurrence(data_dir=args.data_dir)

        # Graph Base On Train_Triplets
        self.graph = nx.MultiDiGraph()
        for h, r, t in self.train_triplets:
            self.graph.add_edge(h, t, relation=r)
        print(self.graph)

    def neighbors_condition(self, ent, rel, direct):
        """
        根据某个实体ent及其相关关系，挑选出与其最相关的（邻居实体 + 关系）
        :param ent:
        :param rel:
        :param direct:
        :return:
        """
        out_edges = []
        score_out = []
        for h, t, attr_dict in self.graph.out_edges(ent, data=True):
            assert ent == h
            out_edges.append((h, attr_dict['relation'], t))
            score_out.append(self.relation_occurrence.rel_occurrences[((rel, direct), (attr_dict['relation'], 0))])
        out_sorted_indices_desc = np.argsort(score_out)[::-1]

        in_edges = []
        score_in = []
        for h, t, attr_dict in self.graph.in_edges(ent, data=True):
            assert ent == t
            in_edges.append((h, attr_dict['relation'], t))
            score_in.append(self.relation_occurrence.rel_occurrences[((rel, direct), (attr_dict['relation'], 1))])
        in_sorted_indices_desc = np.argsort(score_in)[::-1]

        if self.args.neighbor_num <= len(out_edges):
        # return out_edges[out_sorted_indices_desc[:self.args.neighbor_num]]
        elif self.args.neighbor_num <= len(out_edges + in_edges):
            # return out_edges + in_edges[in_sorted_indices_desc[:self.args.neighbor_num - len(out_edges)]]
        else:
            edges = out_edges + in_edges
            random.shuffle(edges)
            return edges

    def neighbors(self):
        """
        :return:
        """
        pass


"""
    数据预处理: 使用TransE的嵌入Embeddings
"""


def TransE_preprocess(args, graph: KnowledgeGraph):
    def load_triplets_with_ids(file_path):
        triplets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            data_num = int(lines[0].strip())
            for line in lines:
                h, r, t = line.strip().split(' ')
                triplets.append((int(h), int(r), int(t)))
            assert data_num == len(triplets), f'{data_num} is not equal to {len(triplets)}'
        return triplets

    def load_ent_or_rel_with_id(file_path):
        ent2id = dict()
        id2ent = dict()
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num = int(lines[0].strip())
            for line in lines:
                ent, id = line.strip().split('\t')
                id = int(id)
                ent2id[ent] = id2ent
                id2ent[id] = ent
            assert num == len(ent2id)
        return ent2id, id2ent

    TransE_dir = os.path.join(args.data_dir, args.kge_model)  # DIFT/FB15K237/TransE || DIFT/WN18RR/TransE

    ent2name = graph.ent2name

    valid_triplets = load_triplets_with_ids(os.path.join(TransE_dir, 'valid2id.txt'))
    # assert len(valid_triplets) == len(graph.valid_triplets)
    test_triplets = load_triplets_with_ids(os.path.join(TransE_dir, 'test2id.txt'))
    # assert len(test_triplets) == len(graph.test_triplets)

    ent2id, id2ent = load_ent_or_rel_with_id(os.path.join(TransE_dir, 'entity2id.txt'))
    # assert len(id2ent) == len(graph.id2ent)
    rel2id, id2rel = load_ent_or_rel_with_id(os.path.join(TransE_dir, 'relation2id.txt'))
    # assert len(id2rel) == len(graph.rel2name)

    # [!!!important]
    entity_embeddings_path = os.path.join(TransE_dir, 'entity_embeddings.pt')
    if not os.path.exists(entity_embeddings_path):
        ent_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embeds_ent.npy')))
        # assert ent_embeds.shape[0] == len(graph.id2ent)
        new_ent_emb = torch.zeros_like(ent_embeds)
        for idx in range(ent_embeds.shape[0]):
            ent = id2ent[idx]
            new_ent_emb[graph.ent2id[ent]] = ent_embeds[idx]  # ???
        assert new_ent_emb.shape[0] == len(graph.ent2id)
        torch.save(new_ent_emb, entity_embeddings_path)

    # [!!!important]
    query_embeddings_path = os.path.join(TransE_dir, 'query_embeddings.pt')
    if not os.path.exists(query_embeddings_path):
        h_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embed_h.npy')))
        r_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embed_r.npy')))
        t_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embed_t.npy')))
        triplets = valid_triplets + test_triplets
        assert h_embeds.shape[0] == r_embeds.shape[0] == t_embeds.shape[0] == len(triplets)

        query_embeddings = torch.zeros(2 * len(triplets), h_embeds.shape[-1])
        idx = 0
        for i in range(len(triplets)):
            query_embeddings[idx] = t_embeds[i] - r_embeds[i]
            query_embeddings[idx + 1] = h_embeds[i] + r_embeds[i]
            idx += 2
        torch.save(query_embeddings, query_embeddings_path)

    query_embeds = torch.load(query_embeddings_path, map_location='cpu')
    ent_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embeds_ent.npy')))
    rel_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embeds_rel.npy')))
    head_ranks = np.load(os.path.join(TransE_dir, 'head_rank.npy'))
    head_topks = np.load(os.path.join(TransE_dir, 'head_topk.npy'))
    head_topks_scores = np.load(os.path.join(TransE_dir, 'head_topk_score.npy'))
    tail_ranks = np.load(os.path.join(TransE_dir, 'tail_rank.npy'))
    tail_topks = np.load(os.path.join(TransE_dir, 'tail_topk.npy'))
    tail_topks_scores = np.load(os.path.join(TransE_dir, 'tail_topk_score.npy'))

    data = []
    triplets = valid_triplets + test_triplets
    for idx, (h, r, t) in enumerate(graph.valid_triplets + graph.test_triplets):
        h2id, r2id, t2id = triplets[idx]
        assert all(query_embeddings[2 * idx] == ent_embeds[t2id] - rel_embeds[r2id])
        assert all(query_embeddings[2 * idx + 1] == ent_embeds[h2id] + rel_embeds[r2id])

        tail_topk = [id2ent[e_idx] for e_idx in tail_topks[idx].toList()][:args.topK]  # 当前样本的topK个尾实体url
        tail_topk_scores = [score * 1e-5 for score in
                            tail_topks_scores[idx].toList()[:args.topK]]  # 当前样本的topK个尾实体的得分
        tail_rank = int(tail_ranks[idx])  # 当前样本真实尾实体在预测排序中的排名
        tail_topk_names = [ent2name[ent] for ent in tail_topks]  # 当前样本的topK个尾实体名称
        tail_entity_ids = [graph.ent2idx[ent] for ent in tail_topks]  # 当前样本的topK个尾实体对应KnowledgeGraph的id

        head_topk = [id2ent[e_idx] for e_idx in head_topks[idx].toList()][:args.topK]  # 当前样本的topK个头实体url
        head_topk_scores = [score * 1e-5 for score in
                            head_topks_scores[idx].toList()[:args.topK]]  # 当前样本的topK个头实体的得分
        head_rank = int(head_ranks[idx])  # 当前样本真实头实体在预测排序中的排名
        head_topk_names = [ent2name[ent] for ent in head_topks]  # 当前样本的topK个头实体名称
        head_entity_ids = [graph.ent2idx[ent] for ent in head_topks]  # 当前样本的topK个头实体对应KnowledgeGraph的id

        head_prediction = {
            'triplet': (h, r, t),
            'inverse': True,
            'topk_ents': head_topk,
            'topk_names': head_topk_names,
            'topk_scores': head_topk_scores,
            'rank': head_rank,
            'query_id': 2 * idx,
            'entity_ids': head_entity_ids
        }

        tail_prediction = {
            'triplet': (h, r, t),
            'inverse': False,
            'topk_ents': tail_topk,
            'topk_names': tail_topk_names,
            'topk_scores': tail_topk_scores,
            'rank': tail_rank,
            'query_id': 2 * idx + 1,
            'entity_ids': tail_entity_ids
        }

        data.append(tail_prediction)
        data.append(head_prediction)
    valid_output = data[:len(valid_triplets) * 2]
    test_output = data[len(valid_triplets) * 2:]

    assert len(graph.valid_triplets) == len(valid_output) // 2
    assert len(graph.test_triplets) == len(test_output) // 2
    return valid_output, test_output


"""
    创建 train/valid/test 数据集
"""


def divide_valid(args, data: list):
    # 划分训练集和测试集
    random.shuffle(data)
    valid_data = data[:int(len(data) * 0.1)]
    train_data = data[int(len(data) * 0.1):]

    # 计算实体的置信度得分
    score_list = []
    for item in train_data:
        if item['rank'] in args.topK:
            score_list.append(100 * item['score'][item['rank'] - 1] + 1 / item['rank'])
        else:
            score_list.append(1 / item['rank'])

    weights = np.array(score_list)
    threshold = args.threshold
    indices = np.where(weights > threshold)[0]
    print("keeped train", len(indices) / len(train_data))

    filter_train = []
    for i in range(len(train_data)):
        if i in indices:
            filter_train.append(train_data[i])
    print(f'train: {len(filter_train)} valid: {len(valid_data)}')
    return filter_train, valid_data


def make_prompt(input_dict, graph: KnowledgeGraph):
    """
    :param input_dict: 是一个字典, {triplet, inverse, topk_ents, topk_names, topk_scores, rank, query_id, entity_ids}
    其中,query_id, entity_id是知识注入时所需要的额外数据, 需要填充的数据包包括input, output
    :param graph:
    :return:
    """
    args = graph.args

    tail_prediction = not input_dict['reverse']
    h, r, t = input_dict['triplets']

    ent2name, ent2desc, rel2name, = graph.ent2name, graph.ent2desc, graph.rel2name
    h_name, h_desc = ent2name[h], ent2desc[h]
    r_name = rel2name[r]
    t_name, t_desc = ent2name[t], ent2desc[t]

    if args.shuffle_candidates:
        topk_ents = input_dict['topk_ents']
        choices = deepcopy(topk_ents)
        random.shuffle(choices)
        entity_ids = [graph.ent2id[ent] for ent in choices]
        input_dict['entity_ids'] = entity_ids
        choices = [graph.ent2name[ent] for ent in choices]
    else:
        choices = input_dict['topk_names']
    input_dict['choices'] = choices  # ???可以不要choice直接在topk_names上进行修改

    if args.add_special_tokens:
        try:
            choices = [ent_name + '[ENTITY]' for ent_name in choices]
        except:
            print(input_dict)
            print(choices)
            exit(0)


def make_dataset_mp():
    pass
