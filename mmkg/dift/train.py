import os
import argparse
from utils import ModelArguments, DataArguments, TrainingArguments, EvaluationArguments, GenerationArguments, get_logger

from transformers import HfArgumentParser, set_seed, GenerationConfig, AutoTokenizer


def train():
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, _ = hf_parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    assert args.model_class in ['KGELLama', 'KGEBert']
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



if __name__ == '__main__':
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    # train()
    tokenizer = AutoTokenizer.from_pretrained('models--TheBloke--Llama-2-7B-fp16', use_fast=True)
    print(tokenizer.pad_token)
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.pad_token)
