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

# 11. torch.sparse (coo_tensor、 mm)
x = torch.sparse_coo_tensor(indices=[[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2]], values=[1, 2, 5, 6, 7, 9], size=(3, 3))
y = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.long)
z = torch.sparse.mm(x, y)
print(z)


# 12. 稀疏矩阵自定义梯度下降
class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, size, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, size)
        ctx.save_for_backward(a, b)
        ctx.N = size[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


# 示例数据
indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
values = torch.tensor([2.0, 3.0], requires_grad=True)
shape = (3, 3)
b = torch.randn(3, 2, requires_grad=True)

# 创建并执行前向和反向传播
spmm = SpecialSpmm()
output = spmm(indices, values, shape, b)

# 对output求和，然后执行backward
loss = output.sum()
loss.backward()
print(output)

# 13. torch.permute
x = torch.randn((3, 1, 4))
print(x)
y = torch.permute(x, [1, 0, 2])
print(y)

# 14. nn.functional.softmax
x = torch.tensor([[[[1, 1, 1, 1], [2, 2, 2, 2]]]], dtype=torch.float)
print(nn.functional.softmax(x, dim=-1))

# 15. torch.mul 逐元素乘
x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])  # [2, in_feat]
y = torch.tensor([[6, 7, 8, 9]])  # [1, out_feat]
print(x.mul(y))

# 16. torch.instanceNorm1d
# 同一batch每个通道每个样本单独计算均值和方差
x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [1, 2, 3]]], dtype=torch.float)  # [2, 2, 3]
instance_norm = nn.InstanceNorm1d(num_features=2, eps=0, affine=False)
print(instance_norm(x))
# layer_norm
# 可以指定形状
x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float)  # [2, 2, 3]
layer_norm = nn.LayerNorm(normalized_shape=(2, 2, 3), eps=0, elementwise_affine=False, bias=False)
print(layer_norm(x))
# batch_norm
# 不同batch每个通道每个样本单独计算均值和方差
x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float)
batch_norm = nn.BatchNorm1d(num_features=2, eps=0, affine=False)
print(batch_norm(x))

# 17. F.one_hot
import torch.nn.functional as F

labels = F.one_hot(torch.arange(start=0, end=4, dtype=torch.int64), num_classes=4)
print(labels)

# 18. pad_sequence
from torch.nn.utils.rnn import pad_sequence

seq1 = torch.tensor([1, 2, 3])
seq2 = torch.tensor([4, 5])
seq3 = torch.tensor([6])

padded = pad_sequence([seq1, seq2, seq3], batch_first=True, padding_value=0)
print(padded)

# 19. torch.nonzero
target = 15
inputs = torch.tensor([[1, 2, 15, 3], [3, 15, 4, 5], [5, 6, 4, 15]])
print(torch.nonzero(inputs == target))

# 20. torch三维替换
x = torch.tensor([[1, 2, 3], [2, 3, 4], [2, 3, 1]])
query = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
pred = nn.Embedding(5, 5)
print(pred)
res = pred(x)
print(res)

idx = torch.nonzero(x == 2)
res[idx[:, 0], idx[:, 1]] = query
print(res)
