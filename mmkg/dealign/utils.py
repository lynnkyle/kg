import numpy as np
import torch


def pairwise_distances(x, y=None):
    """
    :param x:
    x_norm: [N, 1]
    :param y:
    y_norm: [1, M]
    :return:
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    distance = x_norm + y_norm - 2 * torch.mm(x, y.t())
    return torch.clamp(distance, 0.0, np.inf)
