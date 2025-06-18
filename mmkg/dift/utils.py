import logging
import os

import torch
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from peft.tuners.lora import LoraLayer
import transformers
from transformers import BitsAndBytesConfig


def get_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'logs.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_accelerate_model(args, config, pretrained_model_class):
    """
        启用模型量化 模型高效微调peft
    """
    print(f'Loading base model {args.model_name_or_path}...')
    # device_map = 'auto' if os.environ.get('LOCAL_RANK') is None else {'': int(os.environ.get('LOCAL_RANK', 0))}
    # 模型量化
    if args.use_quant:
        compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        model = pretrained_model_class.from_pretrained(args.model_name_or_path, config=config, device_map='auto',
                                                       quantization_config=BitsAndBytesConfig(
                                                           load_in_4bit=args.bits == 4,
                                                           load_in_8bit=args.bits == 8,
                                                           llm_int8_threshold=6.0,
                                                           llm_int8_has_fp16_weight=False,
                                                           bnb_4bit_compute_dtype=compute_dtype,
                                                           bnb_4bit_use_double_quant=args.double_quant,
                                                           bnb_4bit_quant_type=args.quant_type
                                                       ),
                                                       torch_dtype=compute_dtype)
    else:
        model = pretrained_model_class.from_pretrained(args.model_name_or_path, config=config, low_cpu_mem_usage=True,
                                                       device_map='auto')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    # 模型微调(高效微调Lora)
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_quant)
        print(f'Adding LoRA modules...')
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model


class SavePeftModelCallback(transformers.TrainerCallback):

    def save_model(self, args, state, kwargs):
        print("Saving PEFT checkout...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint,
                                             'adapter_model')  # evaluation_strategy == 'no'
        else:
            checkpoint_folder = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')

        peft_model_path = os.path.join(checkpoint_folder, 'adapter_model')
        kwargs['model'].save_pretrained(peft_model_path)

        # 节省磁盘空间，只保留你需要保存的 adapter 权重或特定内容（如 LoRA 的 kge 权重）
        for file_name in os.listdir(checkpoint_folder):
            if 'kge' in file_name:
                continue
            file_path = os.path.join(checkpoint_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
