import os
import argparse
from typing import Optional
from dataclasses import dataclass, field

import torch.cuda

from data import make_data_module
from model import EmbeddingModel, KGELlama
from utils import get_logger, get_accelerate_model, SavePeftModelCallback
from transformers import HfArgumentParser, set_seed, GenerationConfig, AutoTokenizer, AutoConfig, LlamaForCausalLM, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    model_class: str = field(
        default='KGELlama',
        metadata={'help': 'KGELLama(decoder-only) | KGEBert'}
    )
    model_name_or_path: Optional[str] = field(
        default='models--TheBloke--Llama-2-7B-fp16',
        metadata={'help': 'LLM Path'}
    )
    kge_model: Optional[str] = field(
        default='MyGo',
        metadata={'help': 'TransE | CoLE | SimKGC'}
    )
    embedding_dim: int = field(
        default=768,
        metadata={'help': 'Embedding Dimension For KGEModel'}
    )


@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={'help': 'Fine Turn On Which Dataset'})
    train_path: str = field(default=None, metadata={'help': 'Path For Train File'})
    eval_path: str = field(default=None, metadata={'help': 'Path For Valid File'})
    test_path: str = field(default=None, metadata={'help': 'Path For Test File'})
    source_max_len: int = field(default=2048, metadata={'help': 'Maximum Source Sequence Length'})
    target_max_len: int = field(default=64, metadata={'help': 'Maximum Target Sequence Length'})


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    # 大模型参数 微调与量化(高精度参数转换为低精度参数 float32-int4)
    full_finetune: bool = field(default=False, metadata={'help': 'FineTurn Entire Model Without Adapter'})
    use_quant: bool = field(default=False, metadata={'help': 'Use Quantized Model During Training Or Not'})
    double_quant: bool = field(default=True, metadata={'help': 'Compress Statistics Through Double Quantization.'})
    quant_type: str = field(default='nf4', metadata={'help': 'Quantization Type To Use fp4 Or nf4'})
    bits: int = field(default=4, metadata={'help': 'Bit Of Compress Statistics'})

    # KGEModel参数
    do_train: bool = field(default=True, metadata={'help': 'Train Or Not'})
    do_eval: bool = field(default=True, metadata={'help': 'Eval Or Not'})
    output_dir: str = field(default='./output', metadata={'help': 'Output Dir For Logs And Checkpoints'})

    dataloader_num_workers: int = field(default=8, metadata={'help': 'Treads To Load Data'})
    num_train_epoches: int = field(default=3, metadata={'help': 'Total Epoch(1-3)'})
    per_device_train_batch_size: int = field(default=1, metadata={'help': 'Per Device Training Batch Size'})
    gradient_accumulation_steps: int = field(default=16, metadata={'help': 'Gradient Accumulation Steps'})

    optim: str = field(default='paged_adamw_32bit', metadata={'help': 'Optimization Method'})
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
    full_finetune: bool = field(default=False, metadata={'help': 'FineTurn Entire Model Without Adapter'})


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


def train():
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, _ = hf_parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    assert args.model_class in ['KGELlama', 'KGEBert']
    if args.kge_model == 'TransE':
        args.embedding_dim = 250
    set_seed(args.seed)

    os.makedirs(args.output_dir)
    logger = get_logger(args.output_dir)
    logger.info(vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    if args.model_class == 'KGELlama':
        tokenizer.add_tokens(['[QUERY]', '[ENTITY]'])

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = get_accelerate_model(args, model_config, LlamaForCausalLM)
    model.config.use_cache = False

    if args.model_class == 'KGELlama':
        llm_config = model.config
        kge_embedding_dir = os.path.join(args.dataset, args.kge_model)
        embed_model = EmbeddingModel(kge_embedding_dir, args.embedding_dim, 1024, llm_config.hidden_size,
                                     llm_config.hidden_act)
        model = KGELlama(tokenizer, model, embed_model)

    data_module = make_data_module(args, tokenizer, logger)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)  # 训练完成后，只保存PEFT adapter权重，而不是整个模型

    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_metrics('train', metrics)  # 保存训练过程中的评估指标(损失、Hit1...)
        trainer.save_state()  # 保存训练过程的中间状态(优化器状态、学习率调度器状态、当前的epoch, step)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    torch.cuda.set_device(0)
    train()
