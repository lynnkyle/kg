import os

import numpy as np
import torch


def load_text(data_dir, tokenizer, max_seq_len):
    """
    加载实体名称、实体描述、关系名称、文本描述(50个token)
    :param data_dir:
    :param tokenizer:
    :param max_seq_len:
    :return:
    """
    pass


class KnowledgeGraph(object):
    def __init__(self, args):
        self.ent2name, self.ent2desc, self.rel2name = load_text(args.data_dir, tokenizer, args.max_seq_len)
        pass


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
    entity_embeddings_path = os.path.join(TransE_dir, 'entity_embeddings.txt')
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
