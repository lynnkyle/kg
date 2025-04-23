import torch
from torch import nn


def goldE(self, head, hb, relation_v, relation_p, relation_q, relation_f, relation_c, r_bias, theta, e_theta, tail, tb,
          beta, mode):
    r_p = torch.chunk(relation_p, self.p, 3)
    if self.q != 0:
        r_q = torch.chunk(relation_q, self.q, 3)
        r_f = torch.chunk(relation_f, self.f, 3)

    # ??? head_p为什么是四维的(p部分用于双曲线部分计算, q部分用于球面空间计算)
    head_p = head[:, :, :, :self.p]
    tail_p = tail[:, :, :, :self.p]
    head_q = head[:, :, :, self.p:]
    tail_q = tail[:, :, :, self.p:]

    # 偏置拆分 + 非线性映射(超曲空间投影split_pq)
    h_bias, r_bias, t_bias = torch.chunk(r_bias, 3, -1)
    head_x = self.split_pq(head_p, relation_c)
    tail_x = self.split_pq(tail_p, relation_c)

    # 球面部分的旋转操作(镜面反射)
    for i in range(self.q):
        head_q = head_q - 2 * (head_q * r_f[i]).sum(dim=-1, keepdim=True) * r_f[i]
    # 双曲空间的映射部分

    # p部分再进行旋转变换
