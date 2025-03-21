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

# 3. torch.stack [!!!important 先新增维度再进行堆叠]
x = torch.tensor([[0, 1, 2], [3, 4, 5]])  # [2, 3]
y = torch.tensor([[4, 5, 6], [7, 8, 9]])  # [2, 3]
z = torch.stack((x, y), dim=0)  # [2, 2, 3]
print(z)
z = torch.stack((x, y), dim=1)  # [2, 2, 3]
print(z)
z = torch.stack((x, y), dim=2)  # [3, 3, 2]
print(z)

# 4. torch.nn.functional.softmax
x = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float)
y = torch.nn.functional.softmax(x, dim=1)
print(y)

# 5. torch.sum
x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float)
y = torch.sum(x, dim=0)  # [1, 1] -> [1] 降维
print(y)

# 6. *
x = torch.tensor([[[0, 1, 2], [2, 3, 4]], [[2, 3, 4], [0, 1, 2]]], dtype=torch.float)  # [2, 2, 3]
y = torch.tensor([[3, 4], [5, 6]])
print(y.unsqueeze(dim=-1))

# 7. linear
x = torch.randn(10, 128)
linear = nn.Linear(128, 1)
print(linear(x))

# 8. norm (dim=0, 按行规约, 不同行相同列维度计算) (dim=1, 按列规约, 不同列相同行维度计算)
x = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float)
print(x.norm(dim=0))
print(x.sum(dim=0, keepdim=True))

# 9. flatten
x = torch.randn(1, 15)
y = torch.randn(20, 1)
print(x.shape)
print(y.shape)
x = x.flatten()
y = y.flatten()
print(x.shape)
print(y.shape)

# 10. torch.sparse (coo_tensor、 mm)
x = torch.sparse_coo_tensor(indices=[[0, 0, 1, 1], [0, 1, 0, 1]], values=[1, 2, 3, 4], size=(3, 3))
y = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
z = torch.sparse.mm(x, y)
print(z)
