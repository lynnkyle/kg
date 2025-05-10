"""
    大模型微调前的准备工作
    LoRA(Low Rank Adaption Adapter适配层)
    lora_r: lora的秩
    lora_alpha: lora的缩放系数
    lora_dropout: lora的dropout参数
    lora_target_modules: 在哪些模块中插入LoRA层
"""
import os
from transformers import LlamaForCausalLM, LlamaTokenizerFast


def train(
        # model/data/output params
        base_model="", data_path="", output_dir="",
        # training hyperparams
        batch_size=16, micro_batch_size=16, num_epochs=2,
        learning_rate=3e-4, cutoff_len=512, val_set_size: int = 0,
        # lora hyperparams
        lora_r=16, lora_alpha=16, lora_dropout=0.05,
        lara_target_modules=["q_proj", "v_proj"], num_prefix=1,
):
    # 打印训练配置参数
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
        )

    # 加载预训练模型
    model = LlamaForCausalLM.from_pretrained(base_model)
    # 加载分词器
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model)
    print(model.config)
    print(tokenizer.init_kwargs)


if __name__ == '__main__':
    train(base_model="models--TheBloke--Llama-2-7B-fp16")
