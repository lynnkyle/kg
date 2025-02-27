import torch
from torch import nn

# 1. Transformer
"""
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
encoder = nn.TransformerEncoderLayer(d_model=8, nhead=4, batch_first=True)  # [!!!important] batch_first, transformer默认格式 [seq_len, batch_size, features]
output = encoder(inputs, src_key_padding_mask=(padding == 1))
print(output.shape) # [batch_size, seq_len, emb_dim]
