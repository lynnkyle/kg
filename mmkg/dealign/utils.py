import numpy as np
import torch
import scipy


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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(sparse_mx.row, sparse_mx.col)).astype(np.int64)
    values = torch.from_numpy(sparse_mx.data)
    size = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor()


def get_adjr(ent_size, triples, norm=False):
    print('getting a sparse tensor r_adj...')
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        M[(tri[0], tri[2])] += 1
    idx, val = [], []
    for (fir, sec) in M:
        idx.append((fir, sec))
        idx.append((sec, fir))
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])
    for i in range(ent_size):
        idx.append((i, i))
        val.append(1)

    if norm:
        idx = np.array(idx, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = scipy.sparse.coo_matrix((val, (idx[:, 0], idx[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(idx), torch.FloatTensor(val), torch.Size([ent_size, ent_size]))
        return M
