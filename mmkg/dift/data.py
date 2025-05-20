import json
import torch
from torch.utils.data import Dataset


class KGDataModule(object):
    def __init__(self, args, tokenizer, logger=None):
        self.args = args
        self.tokenizer = tokenizer

        train_example = json.load(open(args.train_file, 'r', encoding='utf-8'))
        valid_example = json.load(open(args.eval_file, 'r', encoding='utf-8'))
        test_example = json.load(open(args.test_file, 'r', encoding='utf-8'))

        self.train_dataset = KGDataset(train_example)
        self.valid_dataset = KGDataset(valid_example)
        self.test_dataset = KGDataset(test_example)


class KGDataset(Dataset):
    def __init__(self, example):
        self.data = example
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


class KGDataCollator(object):
    def __init__(self, args, tokenizer, source_max_length, target_max_length):
        self.args = args
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __call__(self, instances):
        sources = [f"{self.tokenizer.bos_token} {example['input']}" for example in instances]
        targets = [f"{example['output']} {self.tokenizer.eos_token}" for example in instances]

        # Tokenize(source：输入,含提示词, target:标签)
        tokenized_sources = self.tokenizer(sources, max_length=self.source_max_length,
                                           truncation=True, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, max_length=self.target_max_length, truncation=True,
                                           add_special_tokens=False)
        source_input_ids = tokenized_sources["input_ids"]
        target_input_ids = tokenized_targets["input_ids"]

        # LLAMA Input Construction

