import os
import pickle

import numpy as np


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
    vis_feat, ent_wo_vis, ent_w_vis = load_vis(logger, ENT_NUM, vis_vec_path, ills)
    logger.info(f"visual feature shape:{vis_feat.shape}")
    logger.info(f"[{len(ent_wo_vis)}] entities have no image")

    # if args.word_embedding == "glove":
    #     word2vec_path = os.path.join(args.data_path, 'pkls', args.data_split + f'_GA_word_vectors.pkl')
    # elif args.word_embedding == "bert":
    #     pass
    # else:
    #     raise Exception("word embedding type error")

    train_ill = np.array(ills[:int(len(ills) * args.data_rate)], dtype=np.int32)
    test_ill = np.array(ills[int(len(ills) * args.data_rate):], dtype=np.int32)

    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))
    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))

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
