import torch

# 1.返回张量中所有非零元素的索引
x = torch.randint(low=0, high=2, size=(3, 3))
print(x)
print(torch.nonzero(x))

# 2.tensor(data=)和Tensor(size=)的区别
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("x===>", x)
y = torch.Tensor(size=(3, 3))
print("y===>", y)
z = torch.Tensor(1, 1, 3)
print("z===>", z)

# 3. unsqueeze操作、cat操作
ent_emb = torch.tensor([[1, 2, 1], [3, 2, 1], [4, 3, 2], [2, 5, 4]])
rel_emb = torch.tensor([[5, 3, 2], [4, 3, 3], [1, 2, 1], [3, 2, 1]])
emb_ent = torch.unsqueeze(ent_emb, dim=1)
emb_rel = torch.unsqueeze(rel_emb, dim=1)
emb = torch.cat((emb_ent, emb_rel), dim=1)
print(emb)

# 4.conv1d的卷积操作
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

# 5.batch_norm [batch, seq_len, embedding_dim]
#   形式一: batch_norm
#   batch_norm针对的是通道embedding_dim
#   layer_norm针对的是样本
x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [7, 8, 9]]], dtype=torch.float32)
y = x.permute(0, 2, 1)  # [batch, embedding_dim, seq_len]
norm = nn.BatchNorm1d(num_features=3, eps=0, affine=False)
print(norm(y))
#   形式二: batch_norm
norm = nn.BatchNorm1d(2)
x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
print(norm(x))

# conv1d
ent_emb = torch.tensor([[2, 1], [1, 2], [2, 1]], dtype=torch.float32)
rel_emb = torch.tensor([[1, 2], [2, 1], [1, 2]], dtype=torch.float32)
ent_emb = torch.unsqueeze(ent_emb, dim=1)
rel_emb = torch.unsqueeze(rel_emb, dim=1)
emb = torch.cat((ent_emb, rel_emb), dim=1)  # [batch_norm, in_channel, emb_dim]
print("emb===>", emb)
conv1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, padding=1,
                  bias=False)  # [out_channel, in_channel, emb_dim]
print(conv1.weight.data)
print("emb===>", conv1(emb))

# batch_norm_1d 不同batch，同一通道(针对2D形状Tensor[batch_size, dim])
tensor = torch.tensor([[[1, 2], [2, 3]], [[2, 3], [3, 4]]], dtype=torch.float32)
norm = nn.BatchNorm1d(num_features=2, eps=0, affine=False)
print(norm(tensor))

x = torch.tensor([[1, 4], [3, 2]], dtype=torch.float32)
print(norm(x))

# layer_norm 同一batch, 不同通道(同一样本) [batch_size,seq_len,emb_dim]
x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [1, 2, 3]]], dtype=torch.float32)
norm = nn.LayerNorm(normalized_shape=3, eps=0, bias=False)
print(norm(x))

y = torch.tensor([[[[1, 2], [3, 4]]], [[[3, 4], [1, 2]]]], dtype=torch.float32)
norm = nn.LayerNorm(normalized_shape=(2, 2), eps=0, bias=False)
print(norm(y))

# 6. chunk操作
x = torch.randn((6, 4))
print(x)
y1, y2 = x.chunk(2, dim=1)
print(y1)
print(y2)

# 7. torch乘法
pi = 3.14159265358979323846
x = torch.randn(6, 4).to("cuda:0")
print(x)
print(x * pi)

# 8. index_select
# head = torch.index_select()

# 9. score
x = torch.tensor(
    [[[[1, 1, 1, 1]], [[1, 1, 1, 1]]], [[[1, 1, 1, 1]], [[1, 1, 1, 1]]]],
    dtype=torch.float)  # [2, batch_size-2, 1, embed_dim-4]
print(x.shape)
x = x.norm(dim=0)
print(x)

x = torch.tensor([[[1, 2], [3, 4]], [[2, 3], [4, 5]]], dtype=torch.float)
y = torch.norm(x, dim=0)
print(y)

# 10. zeros
x = torch.zeros(3, 5)  # 等价于torch.zeros((3, 5))
print(x)

# 11. full
x = torch.full((3, 1), False)
print(x)

# 12. tile 沿指定的维度对张量进行重复
x = torch.randn(1, 1, 1)
print(x)
y = torch.tile(x, (2, 1, 1))
print(y)

# 13. inner内积 *逐元素乘
x = torch.tensor([1, 2, 3])
y = torch.tensor([2, 4, 6])
z = torch.inner(x, y)
print(z)

# 14. fill full
a = torch.zeros(4, 3)
x = torch.fill(a, False)
print(x)
y = torch.full((4, 3), False)
print(y)

# 15. 学习率调度器
# 1). WarmRestarts余弦退火
from torch.optim import lr_scheduler
# lr_scheduler.CosineAnnealingWarmRestarts()
# 2). 自定义余弦退火
# lr_scheduler.StepLR(, step_size=5, gamma=0.1)

# 16. 梯度裁剪器

# 17. default_collate
from torch.utils.data.dataloader import default_collate

res = default_collate([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)])
print(res)
# 18 torch.max
x = torch.tensor([[3, 424, 24, 2], [32, 323, 4, 2]])
print(torch.max(x, dim=1)[1])
