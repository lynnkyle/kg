import os
import pickle
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch


class EADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(logger, args):
    KGs, non_train, train_ill_dataset, test_ill_dataset = load_eva_data(logger, args)
    return KGs, non_train, train_ill_dataset, test_ill_dataset


def load_eva_data(logger, args):
    file_dir = os.path.join(args.data_path, args.data_choice, args.data_split)
    lang_list = [1, 2]
    ent2id, ills, triples, r_hs, r_ts, ids = read_raw_data(file_dir, lang_list)
    e1 = os.path.join(file_dir, 'ent_ids_1')
    e2 = os.path.join(file_dir, 'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(ent2id)
    REL_NUM = len(r_hs)
    np.random.shuffle(ills)

    """
        获取图片数据
    """
    data_prefix = ""
    vis_vec_path = os.path.join(args.data_path, 'pkls', args.data_split + f'_GA_id_img_feature_dict{data_prefix}.pkl')
    assert os.path.exists(vis_vec_path)

    train_ill = np.array(ills[:int(len(ills) * args.data_rate)], dtype=np.int32)
    test_ill = np.array(ills[int(len(ills) * args.data_rate):], dtype=np.int32)

    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))
    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))

    logger.info(f"#left_entity : {len(left_ents)}, #right_entity: {len(right_ents)}")
    logger.info(
        f"#left entity not in train set: {len(left_non_train)}, #right entity not in train set: {len(right_non_train)}")

    """
        获取实体的视觉模态
    """
    vis_feat, ent_wo_vis, ent_w_vis = load_vis(logger, ENT_NUM, vis_vec_path, ills)
    logger.info(f"visual feature shape:{vis_feat.shape}")
    logger.info(f"[{len(ent_wo_vis)}] entities have no image")

    # if args.word_embedding == "glove":
    #     word2vec_path = os.path.join(args.data_path, 'pkls', args.data_split + f'_GA_word_vectors.pkl')
    # elif args.word_embedding == "bert":
    #     pass
    # else:
    #     raise Exception("word embedding type error")

    """
        获取实体的关系模态
    """
    rel_feat = load_rel(ENT_NUM, triples, 1000)
    logger.info(f"relation feature shape:{rel_feat.shape}")

    """
        获取实体的属性模态
    """
    attr_vec_path_1 = os.path.join(file_dir, 'training_attrs_1')
    attr_vec_path_2 = os.path.join(file_dir, 'training_attrs_2')
    attr_feat = load_attr([attr_vec_path_1, attr_vec_path_2], ENT_NUM, ent2id, 1000)
    logger.info(f"attribute feature shape:{attr_feat.shape}")

    logger.info("-----dataset summary-----")
    logger.info(f"dataset:\t\t {file_dir}")
    logger.info(f"triple num:\t {len(triples)}")
    logger.info(f"entity num:\t {ENT_NUM}")
    logger.info(f"relation num:\t {REL_NUM}")
    logger.info(f"train ill num:\t {train_ill.shape[0]} \t test ill num:\t {test_ill.shape[0]}")
    logger.info("-------------------------")

    name_feat = None
    txt_feat = None

    eval_ill = None
    input_idx = torch.LongTensor(np.arange(ENT_NUM))
    adj = get_adjr(ENT_NUM, triples, norm=True)

    train_ill_Dataset = EADataset(train_ill)
    test_ill_Dataset = EADataset(test_ill)

    return {
        'ent_num': ENT_NUM,
        'rel_num': REL_NUM,
        'vis_feat': vis_feat,
        'ent_wo_vis': ent_wo_vis,
        'ent_w_vis': ent_w_vis,
        'rel_feat': rel_feat,
        'attr_feat': attr_feat,
        'name_feat': name_feat,
        'txt_feat': txt_feat,
        'input_idx': input_idx,
        'adj': adj,
        'train_ill': train_ill,
        'test_ill': test_ill
    }, {
        "left": left_non_train,
        "right": right_non_train,
    }, train_ill_Dataset, test_ill_Dataset


def read_raw_data(file_dir, lang_list=[1, 2]):
    print('loading raw data...')

    def read_file(file_paths):
        tuples = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    params = line.strip("\n").split('\t')
                    tuples.append(tuple(int(x) for x in params))
        return tuples

    def read_dict(file_paths):
        ent2id = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    params = line.strip("\n").split('\t')
                    ent2id[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id, ids

    ent2id, ids = read_dict([file_dir + '/ent_ids_' + str(i) for i in lang_list])
    ills = read_file([file_dir + '/ill_ent_ids'])
    triples = read_file([file_dir + '/triples_' + str(i) for i in lang_list])

    r_hs, r_ts = {}, {}
    for triple in triples:
        h, r, t = triple
        if r not in r_hs:
            r_hs[r] = set()
        r_hs[r].add(h)
        if r not in r_ts:
            r_ts[r] = set()
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return ent2id, ills, triples, r_hs, r_ts, ids


def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            params = line.strip("\n").split('\t')
            ids.append(int(params[0]))
    return ids


def load_vis(logger, ent_num, path, ills):
    vis_dict = pickle.load(open(path, 'rb'))

    vis_np = np.array(list(vis_dict.values()))
    mean = np.mean(vis_np, axis=0)
    std = np.std(vis_np, axis=0)
    vis_emb = np.array(
        [vis_dict[i] if i in vis_dict else np.random.normal(mean, std, vis_np[0].shape[0]) for i in range(ent_num)])
    ent_wo_vis = [i for i in range(ent_num) if i not in vis_dict]
    ent_w_vis = [i for i in range(ent_num) if i in vis_dict]

    all_ent = [i[0] for i in ills] + [i[1] for i in ills]
    ent_w_vis_ill = [i for i in all_ent if i in ent_w_vis]

    logger.info(f"{(100 * len(ent_w_vis)) / ent_num:.2f}% entities have images")
    logger.info(
        f"{(100 * len(ent_w_vis_ill)) / len(all_ent):.2f}% entities in EA(entity align) dataset have images")

    return vis_emb, ent_wo_vis, ent_w_vis


def load_rel(ent_num, triples, topR=1000):
    rel_mat = np.zeros((ent_num, topR), dtype=np.float32)
    rels = np.array(triples)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index = {r: i for i, (r, cnt) in enumerate(top_rels)}
    """
        构造关系特征矩阵
    """
    for tri in triples:
        h = tri[0]
        r = tri[1]
        t = tri[2]
        if r in rel_index:
            rel_mat[h][rel_index[r]] += 1
            rel_mat[t][rel_index[r]] += 1
    return np.array(rel_mat)


def load_attr(fns, ent_num, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                params = line[:-1].strip().split('\t')
                if params[0] not in ent2id:
                    continue
                for i in range(1, len(params)):
                    if params[i] not in cnt:
                        cnt[params[i]] = 1
                    else:
                        cnt[params[i]] += 1

    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]  # 降序排序
    attr2id = {}
    topA = min(1000, len(fre))
    for i in range(topA):
        attr2id[fre[i][0]] = i
    """
        构造属性特征矩阵
    """
    attr = np.zeros((ent_num, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:, -1].strip().split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[attr2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def normalize_adj(mx):
    row_sum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def get_adjr(ent_size, triples, norm=False):
    print('getting a sparse tensor r_adj...')

    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        M[(tri[0], tri[2])] += 1

    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        ind.append((sec, fir))
        val.append(M[fir, sec])
        val.append(M[fir, sec])

    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)

    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size]))
        return M
