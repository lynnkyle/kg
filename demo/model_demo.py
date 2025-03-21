import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

# 1. Transformer
"""
    实验一:
    Encoder编码器不包括位置编码
    Encoder编码器被掩码的位置不会参与计算
"""
embedding = nn.Embedding(num_embeddings=10, embedding_dim=8)
x = torch.tensor([
    [1, 2, 3, 4, 0, 0, 0],  # Sentence 1
    [5, 6, 7, 8, 0, 0, 0],  # Sentence 2
    [9, 1, 1, 7, 2, 0, 0]  # Sentence 3
])
padding = torch.tensor([
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1],
])
inputs = embedding(x)  # [batch_size, seq_len, emb_dim]
encoder = nn.TransformerEncoderLayer(d_model=8, nhead=4,
                                     batch_first=True)  # [!!!important] batch_first, transformer默认格式 [seq_len, batch_size, features]
output = encoder(inputs, src_key_padding_mask=(padding == 1))
print(output.shape)  # [batch_size, seq_len, emb_dim]

"""
    实验二:
    Encoder编码器的输入
    [batch_size, seq_len, embedding_dim]
    d_model参数是embedding_dim, 要求输入的每个token的维度要相同
"""
encoder = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
x = torch.randn((2, 3, 16))
output = encoder(x)
print(output.shape)

"""
    实验三:
    对于三维输入[batch_size, seq_len, emb_dim] seq_len: 句子分词个数
    Transformer输出返回的大小, tensor索引取值, 维度是1时, 会自动降维
"""
x = torch.randn((10, 3, 16))
encoder = TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
y = encoder(x)[:, 0]
print(y.shape)  # shape: [10, 16]
print(x[:, 0].shape)  # shape: [10, 16]

"""
    实验四:
    在model.train模式下, dropout会影响代码复现
"""
# random.seed(0)
# np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(8)
torch.cuda.manual_seed_all(0)
torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
x = torch.randn((10, 3, 256))
encoder_layer = TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=1024, dropout=0.01, batch_first=True)
encoder_1 = TransformerEncoder(encoder_layer, num_layers=1)
# encoder_1.eval()
print(encoder_1(x))
print(encoder_1(x))

