import torch
from torch import nn

# 1.torch.mean var std
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
print(torch.mean(x, dim=0))
print(torch.var(x, dim=0))
print(torch.std(x, dim=0))

# 2.nn.Embedding nn.Parameter
"""
    nn.Embedding 在 GPU 上优化了索引查找, 可高效处理大规模的词嵌入（如 NLP 任务）
    nn.Parameter 是普通张量, 不会优化索引查找, 如果数据量大, 可能会导致计算低效
"""
x = torch.randn(8, 6)
print(x)
y = nn.Embedding.from_pretrained(x).requires_grad_(False)
z = nn.Parameter(x, requires_grad=False)
print(y(torch.tensor([0, 2, 4, 6])))
print(z[[0, 2, 4, 6]])
