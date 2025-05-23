import argparse
from typing import Optional

import torch

"""
    1. 分词器Tokenizer的使用
"""
from transformers import AutoTokenizer, TrainingArguments, Seq2SeqTrainingArguments

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
# import matplotlib.pyplot as plt
#
# nx.draw(g, with_labels=True)
# plt.show()

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
class TrainingArguments(Seq2SeqTrainingArguments):
    # 大模型参数 微调与量化(高精度参数转换为低精度参数 float32-int4)
    train_full_finetune: bool = field(default=False, metadata={'help': 'FineTurn Entire Model Without Adapter'})
    use_quant: bool = field(default=False, metadata={'help': 'Use Quantized Model During Training Or Not'})
    double_quant: bool = field(default=True, metadata={'help': 'Compress Statistics Through Double Quantization.'})
    quant_type: str = field(default='nf4', metadata={'help': 'Quantization Type To Use fp4 Or nf4'})
    bits: int = field(default=4, metadata={'help': 'Bit Of Compress Statistics'})

    # KGEModel参数
    do_train: bool = field(default=True, metadata={"help": 'Train Or Not'})
    do_eval: bool = field(default=True, metadata={"help": 'Eval Or Not'})
    output_dir: str = field(default='./output', metadata={"help": 'Output Dir For Logs And Checkpoints'})

    dataloader_num_workers: int = field(default=8, metadata={'help': 'Treads To Load Data'})
    num_train_epoches: int = field(default=3, metadata={'help': 'Total Epoch(1-3)'})
    per_device_train_batch_size: int = field(default=1, metadata={'help': 'Per Device Training Batch Size'})
    gradient_accumulation_steps: int = field(default=16, metadata={'help': 'Gradient Accumulation Steps'})

    optim: str = field(default="paged_adamw_32bit", metadata={'help': 'Optimization Method'})
    learning_rate: float = field(default=2e-4, metadata={'help': 'Learning Rate'})
    lr_scheduler_type: str = field(default='constant',
                                   metadata={'help': 'Learning Rate Scheduler: Constant, Linear, Cosine'})
    warmup_ratio: float = field(default=0.03, metadata={'help': 'Warmup Ratio'})

    lora_r: int = field(default=64, metadata={'help': 'LoRa R'})
    lora_alpha: float = field(default=16, metadata={'help': 'LoRa Alpha'})
    lora_dropout: float = field(default=0.0, metadata={'help': 'LoRa Dropout'})

    remove_unused_columns: bool = field(default=False, metadata={'help': 'Remove Unused Columns Or Not'})
    report_to: str = field(default='none', metadata={'help': 'Not Use Logger'})


@dataclass
class EvaluationArguments:
    checkpoint_dir: Optional[str] = field(default=None, metadata={'help': 'Checkpoint Dir'})
    eval_full_finetune: bool = field(default=False, metadata={'help': 'FineTurn Entire Model Without Adapter'})


@dataclass
class GenerationArguments:
    # 设置output长度
    max_new_tokens: Optional[int] = field(default=64, metadata={'help': 'Max New Tokens'})
    min_new_tokens: Optional[int] = field(default=1, metadata={'help': 'Min New Tokens'})

    # 设置生成策略(贪心搜索, 不惩罚相似内容)
    do_sample: Optional[bool] = field(default=True, metadata={'help': 'Sample Or Not'})
    num_beams: Optional[int] = field(default=1, metadata={'help': 'Num Beam'})
    num_beam_groups: Optional[int] = field(default=1, metadata={'help': 'Num Beam Groups'})
    penalty_alpha: Optional[float] = field(default=None, metadata={'help': 'Penalty Alpha'})
    use_cache: Optional[bool] = field(default=True, metadata={'help': 'Use Cache Or Not'})

    # 设置词概率处理(增强多样性, 禁止重复)
    temperature: Optional[float] = field(default=1.0, metadata={'help': 'Temperature'})
    top_k: Optional[int] = field(default=50, metadata={'help': 'Top K'})
    top_p: Optional[float] = field(default=0.9, metadata={'help': 'Top p'})
    typical_p: Optional[float] = field(default=1.0, metadata={'help': 'Typical p'})
    diversity_penalty: Optional[float] = field(default=0.0, metadata={'help': 'Diversity Penalty'})
    repetition_penalty: Optional[float] = field(default=1.0, metadata={'help': 'Repetition Penalty'})
    length_penalty: Optional[float] = field(default=1.0, metadata={'help': 'Length Penalty'})
    no_repeat_ngram_size: Optional[int] = field(default=0, metadata={})

    # 设置输出格式
    num_return_sequences: Optional[int] = field(default=1, metadata={'help': 'Num Return Sequences'})
    output_scores: Optional[bool] = field(default=False, metadata={'help': 'Return Output Scores Or Not'})
    return_dict_in_generate: Optional[bool] = field(default=True, metadata={
        'help': 'Return Dict(Sequences、 Scores、 Beam_Indices、 Sequence_Scores) In Generate Or Not'})


import os
from transformers import HfArgumentParser

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

hfparser = HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments, EvaluationArguments, GenerationArguments))
model_args, data_args, training_args, evaluation_args, generation_args, _ = hfparser.parse_args_into_dataclasses(
    return_remaining_strings=True)
print("medel_args==>", model_args)
print("data_args==>", data_args)
print("eval_args==>", evaluation_args)
print("generation_args==>", generation_args)
# print("training_args==>", training_args)

"""
    5.
    AutoModel: 加载模型
    get_peft_model: 加速/包装模型以支持多卡、混合精度等训练加速方式
"""
args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args), **vars(generation_args))
print('args==>', args)

# 1.AutoModel
# llama = AutoModel.from_pretrained('/home/ps/lzy/subaru/models--TheBloke--Llama-2-7B-fp16')
print(llama)

"""
    6.AutoConfig
"""
from transformers import AutoConfig

config = AutoConfig.from_pretrained('/home/ps/lzy/subaru/models--TheBloke--Llama-2-7B-fp16')
print(config.hidden_size)
print(config.hidden_act)
