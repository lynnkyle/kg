"""
    大模型微调前的准备工作
    LoRA(Low Rank Adaption Adapter适配层)
    lora_r: lora的秩
    lora_alpha: lora的缩放系数
    lora_dropout: lora的dropout参数
    lora_target_modules: 在哪些模块["q_proj", "k_proj", "v_proj", "o_proj"]中插入LoRA层
    num_prefix: 原始输入前额外添加N个可学习的token向量
    Llama(大语言模型)
    train_on_inputs: True- 损失计算针对输入、输出之间的内容; False- 损失计算针对输出的内容
    add_eos_token: False- 样本末尾添加eos标记
    group_by_length: False- 长度相近的样本归为一组, 减少填充, 加快训练速度
"""
import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast

from mmkg.subaru.prompter import Prompter


def train(
        # model/data/output params
        base_model=None, data_path="", output_dir="",
        # training hyperparams
        batch_size=16, micro_batch_size=16, num_epochs=2,
        learning_rate=3e-4, cutoff_len=512, val_set_size: int = 0,
        # lora hyperparams
        lora_r=16, lora_alpha=16, lora_dropout=0.05,
        lara_target_modules=["q_proj", "v_proj"], num_prefix=1,
        # llm hyperparams
        train_on_inputs=True, add_eos_token=False, group_by_length=False,
        # wandb params
        resume_from_checkpoint=None, prompt_template_name="alpaca_short",
        kge_model="data/transe.pt", adapter_type="mlp"

):
    # 训练参数配置
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params\n"
            f"model/data/output params: \n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"training hyperparams: \n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size:{val_set_size}\n"
            f"lora hyperparams: \n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lara_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lara_target_modules}\n"
        )
    assert base_model is not None, "Please specify a --base_model, e.g. --base_model='TheBloke/Llama-2-7B-fp16'"
    gradient_accumulation_steps = batch_size // micro_batch_size  # 梯度累计
    device_map = "auto"
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": os.environ.get('WORLD_SIZE', 1)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    prompter = Prompter(prompt_template_name)  # 提示词
    # 加载预训练模型
    model = LlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map=device_map)
    # 加载分词器
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # <unk>
    tokenizer.padding_side = 'left'  # 左侧填充句子

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None
        )
        # 手动添加<eos>: 适用于微调 decoder-only模型
        # add_special_tokens = True: 适用于已有任务模板(bert,t5)
        if result['input_ids'][-1] != tokenizer.eos_token_id and len(
                result['input_ids']) < cutoff_len and add_eos_token:
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)

        result['labels'] = result['input_ids'].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point['instruction'], data_point['input']
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point['instruction'], data_point['input']
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len =


if __name__ == '__main__':
    # train(base_model="models--TheBloke--Llama-2-7B-fp16")
    # print(os.environ.get('LOCAL_RANK', 0))
    print(os.environ.get('WORLD_SIZE'))
