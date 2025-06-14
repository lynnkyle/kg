import numpy as np


def calculate_rank(score, target, filter_list):
    target_score = score[target]
    score[filter_list] = target_score - 1
    rank = np.sum(score > target_score) + np.sum(score == target_score) // 2 + 1
    return rank


def metrics(rank_list):
    mr = np.mean(rank_list)
    mrr = np.mean(1 / rank_list)
    hit10 = np.sum(rank_list < 11) / len(rank_list)
    hit3 = np.sum(rank_list < 4) / len(rank_list)
    hit1 = np.sum(rank_list < 2) / len(rank_list)
    return mr, mrr, hit10, hit3, hit1


def get_rank(score, ent):
    sorted_indices = np.argsort(-score)  # 降序排序
    rank = np.where(sorted_indices == ent)[0][0] + 1  # 找到 ent 的位置（排名从1开始）
    return rank


def get_topK(score, topK):
    indices = np.argsort(-score)  # 负号表示降序
    vals = score[indices]
    return indices[:topK], vals[:topK]
