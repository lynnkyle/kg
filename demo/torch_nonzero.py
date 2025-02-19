import torch

# 返回张量中所有非零元素的索引
x = torch.randint(low=0, high=2, size=(3, 3))
print(x)
print(torch.nonzero(x))
