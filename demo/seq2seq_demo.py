import os

import torch
from datasets import Dataset
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

"""
    Seq2SeqTrainer: 训练的模型必须包含Encoder-Decoder（编码器-解码器）结构, 支持 .generate() 方法并返回包含 .loss() 的输出
"""
