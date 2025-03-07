import numpy as np


def calculate_rank(score, target, filter_list):
    score_target = score[target]
    score[filter_list] = score_target - 1
    rank = np.sum(score > score_target) + np.sum(score == score_target) // 2 + 1
    return rank


def metrics(list_rank):
    mr = np.mean(list_rank)
    mrr = np.mean(1 / list_rank)
    hit10 = np.sum(list_rank < 11) / len(list_rank)
    hit3 = np.sum(list_rank < 4) / len(list_rank)
    hit1 = np.sum(list_rank < 2) / len(list_rank)
    return mr, mrr, hit10, hit3, hit1
