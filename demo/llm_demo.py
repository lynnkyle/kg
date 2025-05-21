from typing import Optional

import torch

"""
    1. 分词器Tokenizer的使用
"""
from transformers import AutoTokenizer, TrainingArguments

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
from transformers import AutoModel

llama = AutoModel.from_pretrained('/home/ps/lzy/subaru/models--TheBloke--Llama-2-7B-fp16')

# embed_tokens 得到token_id对应的嵌入
embeds = llama.embed_tokens.weight
print(embeds)
embedding = llama.embed_tokens(torch.LongTensor([[0, 1, 2, 3]]))
print(embedding)

# forward 得到隐藏层提取的向量, 用于下游任务
res = llama(torch.LongTensor([[0, 1, 2, 3]]))
print(res)

"""
    3. networkx图论与网络分析库, 用于创建、操作和研究复杂网络结构
"""
import networkx as nx

# 创建图
g = nx.Graph()
# g = nx.MultiGraph(g) 多重图
# g = nx.MultiDiGraph(g) 有向多重图
g.add_node(1)
g.add_nodes_from([2, 3])
g.add_edge(1, 2, relation=3)
g.add_edge(2, 3, relation=1)
g.add_edge(3, 1, relation=2)
# g.add_edges_from([(2, 3), (3, 1)])

# 遍历图
print(g)
print("所有节点", g.nodes())
print("所有边", g.edges())
for u, v, data in g.edges(data=True):
    print(f"{u} -> {v}: relation = {data['relation']}")

# 计算最短路径
path = nx.shortest_path(g, source=1, target=3)
print("从1到3的最短路径:", path)

# 可视化图
import matplotlib.pyplot as plt

nx.draw(g, with_labels=True)
plt.show()

"""
    4.HuggingFace参数解析器
"""

from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_class: str = field(
        default='KGELlama',
        metadata={"help": "KGELLama | KGEBert"}
    )
    model_name_or_path: Optional[str] = field(
        default='models--TheBloke--Llama-2-7B-fp16',
        metadata={'help': "LLM Path"}
    )
    kge_model: Optional[str] = field(
        default='MyGo',
        metadata={"help": "KGELlama | KGEBert"}
    )
    embedding_dim: int = field(
        default=768,
        metadata={"help": "Embedding Dimension For KGEModel"}
    )


@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={'help': 'Fine Turn On Which Dataset'})
    train_path: str = field(default=None, metadata={'help': 'Path For Train File'})
    valid_path: str = field(default=None, metadata={'help': 'Path For Valid File'})
    test_path: str = field(default=None, metadata={'help': 'Path For Test File'})
    source_max_len: int = field(default=2048, metadata={'help': 'Maximum Source Sequence Length'})
    target_max_len: int = field(default=64, metadata={'help': 'Maximum Target Sequence Length'})


@dataclass
class TrainingArguments(TrainingArguments):
    # 大模型参数 微调与量化(高精度参数转换为低精度参数 float32-int4)
    full_finetune: bool = field(default=False, metadata={'help': 'FineTurn Entire Model Without Adapter'})
    use_quant: bool = field(default=False, metadata={'help': 'Use Quantized Model During Training Or Not'})
    double_quant: bool = field(default=True, metadata={'help': 'Compress Statistics Through Double Quantization.'})
    quant_type: str = field(default='nf4', metadata={'help': 'Quantization Type To Use fp4 Or nf4'})
    bits: int = field(default=4, metadata={'help': 'Bit Of Compress Statistics'})

    # KGEModel参数
    do_train: bool = field(default=True, metadata={"help": 'Train Or Not'})
    do_eval: bool = field(default=True, metadata={"help": 'Eval Or Not'})
    output_dir: str = field(default='./output', metadata={"help": 'Output Dir For Logs And Checkpoints'})

    num_


from transformers import HfArgumentParser

HfArgumentParser(())
