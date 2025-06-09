import os

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import AutoConfig, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

# 加载分词器, 加载模型
model_path = '/home/ps/lzy/kg/mmkg/dift/models--TheBloke--Llama-2-7B-fp16'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(model_path)

# 加载数据
train_data = [{
    'question': 'How are you?',
    'answer': 'I am fine. Thank you and you.'
}, {
    'question': 'Where are you come from?',
    'answer': 'I come from JinJiang.'
}, {
    'question': 'When can I success?',
    'answer': 'Next year.'
}]
eval_data = [{
    'question': 'Where are you come from?',
    'answer': 'I come from JinJiang.'
}, {
    'question': 'When can I success?',
    'answer': 'Next year.'
}]


def preprocess_data(example):
    return {
        'input': example['question'],
        'target': example['answer']
    }


train_data = Dataset.from_list(train_data).map(preprocess_data)
eval_data = Dataset.from_list(eval_data).map(preprocess_data)


def tokenize(example):
    model_inputs = tokenizer(example['input'], truncation=True, max_length=32, padding='max_length')
    labels = tokenizer(example['target'], truncation=True, max_length=32, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


train_data = train_data.map(tokenize)
eval_data = eval_data.map(tokenize)

# peft 量化+微调
compute_type = torch.bfloat16
model = LlamaForCausalLM.from_pretrained(
    model_path,
    config=config,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # load_in_8bit=True,
        # llm_int8_threshold=6.0,
        # llm_int8_has_fp16_weight=False
    ),
    device_map='auto',
    torch_dtype=compute_type)
setattr(model, 'model_parallel', True)
setattr(model, 'is_parallelizable', True)
config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

#
train_args = TrainingArguments(
    output_dir='./out',
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    bf16=True,
    learning_rate=2e-4,
    optim='paged_adamw_32bit',
    eval_strategy='epoch',
    logging_steps=10,
    report_to=[]
)

trainer = Trainer(
    tokenizer=tokenizer,
    model=model,
    args=train_args,
    train_dataset=train_data,
    eval_dataset=eval_data
)

result = trainer.train()
metrics = result.metrics
trainer.save_metrics('train', metrics)
