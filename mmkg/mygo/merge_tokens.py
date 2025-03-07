import json
from collections import defaultdict, Counter

import torch


def load_ent_map(dataset):
    ent_map = {}
    if dataset == 'FB15K-237' or dataset == 'DB15K' or dataset == 'WN9':
        with open(f'data/{dataset}/entities.txt', 'r') as f:
            lines = f.readlines()
            for step, line in enumerate(lines):
                ent_map[line.strip()] = int(step)
    else:
        with open(f'data/{dataset}/entity2id.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                _, id = line.strip().split(' ')
                ent_map[_] = int(id)
    return ent_map


"""
    一个实体对应多个token
    一个token对应一个特征向量
"""


def get_entity_visual_tokens(dataset, max_num, type='beit'):
    if type == 'beit':
        tokenized_result = json.load(open(f'tokens/{dataset}-visual.json', 'r'))
        token_size = 8192  # ???
    elif type == 'vggan':
        tokenized_result = json.load(open(f'tokens/{dataset}-visual-vggan.json', 'r'))
        token_size = 1024  # ???
    else:
        raise NotImplementedError

    """
        构建tkn2ent字典: 记录每个token相关的entity
        构建ent2tkn字典: 记录每个entity相关的token
    """

    tkn2ent = defaultdict(list)
    for i in range(token_size + 1):
        tkn2ent[i] = []
    for entity in tokenized_result:
        tkn_count = Counter(tokenized_result[entity])
        selected_tkn = tkn_count.most_common(max_num)
        for tkn, _ in selected_tkn:
            tkn2ent[tkn].append(entity)

    ent2id = load_ent_map(dataset)
    tkn_id = list(tkn2ent.keys())
    ent2tkn = defaultdict(list)
    for i in range(len(tkn_id)):
        for entity in tkn2ent[tkn_id[i]]:
            ent2tkn[ent2id[entity]].append(i)

    ent2tkns = []
    ent_key_mask = []
    for i in range(len(ent2id)):
        if ent2tkn[i] != []:
            ent2tkns.append(ent2tkn[i])
            ent_key_mask.append(([False] * max_num))
        else:
            ent2tkns.append([token_size - 1] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(ent2tkns), torch.BoolTensor(ent_key_mask)


def get_entity_textual_tokens(dataset, max_num, type='bert'):
    # if dataset == 'DB15K':
    #     return get_entity_textual_tokens_db15k(dataset, max_num, type)
    tokenized_result = json.load(open(f'tokens/{dataset}-textual.json', 'r'))
    token2ent = defaultdict(list)
    for i in range(30522 + 1):
        pass


if __name__ == '__main__':
    token = get_entity_visual_tokens('MKG-W', max_num=8)
    print(token)
