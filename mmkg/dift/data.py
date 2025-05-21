import copy
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def make_data_module(args, tokenizer, logger=None):
    data_module = KGDataModule(args, tokenizer, logger)
    data_collator = KGDataCollator(args, tokenizer)
    return {
        'train_dataset': data_module.train_dataset,
        'valid_dataset': data_module.valid_dataset,
        'test_dataset': data_module.test_dataset,
        'data_collator': data_collator
    }


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


IGNORE_INDEX = -100


class KGDataCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.source_max_length = args.source_max_length
        self.target_max_length = args.target_max_length

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

        # LLAMA Input(data_dict) Construction
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(source_input_ids, target_input_ids):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            labels.append(
                torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels
        }
        if self.args.model_class == 'KGELlama':
            data_dict['query_ids'] = torch.LongTensor([
                example['query_id'] for example in instances
            ])
            data_dict['entity_ids'] = torch.LongTensor([
                example['entity_id'] for example in instances
            ])
        else:
            raise NotImplementedError
        return data_dict


class KGDataset(Dataset):
    def __init__(self, example):
        self.data = example
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]
