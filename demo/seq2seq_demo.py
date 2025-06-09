import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    LlamaTokenizer,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model

"""
    Seq2SeqTrainer: 训练的模型必须包含Encoder-Decoder（编码器-解码器）结构, 支持 .generate() 方法并返回包含 .loss() 的输出
"""
data = {
    "translation": [
        {'en': 'i love you', 'zh': 'i love you'},
        {'en': 'hello', 'zh': 'hello'},
        {'en': 'Thank you', 'zh': 'no thanks'}
    ]
}

# 1. 加载分词器 / 模型
tokenizer = T5Tokenizer.from_pretrained('/home/ps/lzy/kg/pre_trained/t5-small')
model = T5ForConditionalGeneration.from_pretrained('/home/ps/lzy/kg/pre_trained/t5-small')


# 2. 数据预处理
def preprocess_data(example):
    return {
        'input': f'translate English to Chinese: {example["en"]}',
        'target': example['zh']
    }


raw_dataset = Dataset.from_list(data['translation']).map(preprocess_data)
print(raw_dataset)


def tokenize(example):
    model_inputs = tokenizer(example['input'], max_length=32, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['target'], max_length=32, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


tokenized_data = raw_dataset.map(tokenize)
print(tokenized_data[0])

# 3.训练模型
training_args = Seq2SeqTrainingArguments(
    output_dir='./result',
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=1,
    predict_with_generate=True,  # output.logits: [batch_size, seq_len, vocab_size]
    eval_strategy='no',
    save_strategy='no',
    report_to=[]
)

trainer = Seq2SeqTrainer(
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
    model=model,
    args=training_args
)

train_result = trainer.train()
metrics = train_result.metrics
# 记录训练的结果
trainer.log_metrics('train', metrics)
trainer.save_metrics('train', metrics)
trainer.save_state()

# 4. 模型训练完成
print('training complete!!!')

# 5. 模型预测
data = ["translate English to Chinese: I love you",
        "translate English to Chinese: How are you?",
        "translate English to Chinese: This is a test."]

val_dataset = Dataset.from_dict({'input_texts': data})


def preprocess(example):
    return tokenizer(example['input_texts'], truncation=True, padding='max_length', max_length=32)


val_dataset = val_dataset.map(preprocess)
print(val_dataset[0])

predictions = trainer.predict(val_dataset)

decoded = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
for src, tgt in zip(val_dataset, decoded):
    print(f"Input: {src['input_texts']}")
    print(f"Prediction: {tgt}")
    print("------")

trainer.add_callback()
