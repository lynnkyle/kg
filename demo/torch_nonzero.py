import torch

# 1.返回张量中所有非零元素的索引
x = torch.randint(low=0, high=2, size=(3, 3))
print(x)
print(torch.nonzero(x))

# 2.conv1d的卷积操作
from torch import nn

emb_ent = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
print(emb_ent)
emb_rel = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
print(emb_rel)
emb_stack = torch.cat([emb_ent, emb_rel], dim=1)
print(emb_stack)
print(emb_stack.size())  # [2,4]
conv1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2,
                  bias=False)  # [out_channel, in_channel, kernel_size]
print(conv1.weight.data)
print(conv1(emb_stack))
