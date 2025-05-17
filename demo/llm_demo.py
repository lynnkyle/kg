import torch

from transformers import AutoTokenizer, AutoModel

"""
    1. 分词器Tokenizer的使用
"""
tokenizer = AutoTokenizer.from_pretrained('/home/ps/lzy/subaru/models--TheBloke--Llama-2-7B-fp16')

text = "谁是谁的爹?我是郑皓哲的爹。谁是谁的儿?郑皓哲是我儿。"

# tokenize 分词
res = tokenizer.tokenize(text)
print(res)
# encode 分词并转换成token id
res = tokenizer.encode(text)
print(res)
# decode token id转换成分词
res = tokenizer.decode(res)
print(res)

# call 分词并产生(input_ids、attention_mask、token_type_ids)
res = tokenizer(text)
print(res)

"""
    2. Llama大语言模型的使用
"""
llama = AutoModel.from_pretrained('/home/ps/lzy/subaru/models--TheBloke--Llama-2-7B-fp16')

# embed_tokens 得到token_id对应的嵌入
embeds = llama.embed_tokens.weight
print(embeds)
embedding = llama.embed_tokens(torch.LongTensor([[0, 1, 2, 3]]))
print(embedding)

# forward 得到隐藏层提取的向量, 用于下游任务
res = llama(torch.LongTensor([[0, 1, 2, 3]]))
print(res)
