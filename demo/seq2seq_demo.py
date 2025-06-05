import os
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# 加载数据集
data = load_dataset('../data/opus_books', 'en-fr')
print(data)

# 加载分词器和模型
t5_model = T5ForConditionalGeneration.from_pretrained('/home/ps/lzy/kg/pre_trained/t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('/home/ps/lzy/kg/pre_trained/t5-small')

"""
    Seq2SeqTrainer: 训练的模型必须包含Encoder-Decoder（编码器-解码器）结构, 支持 .generate() 方法并返回包含 .loss() 的输出
"""
