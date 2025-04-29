import os

import numpy as np
import torch
from easydict import EasyDict
from torch.cuda.amp import GradScaler

from mmkg.dealign.data import load_data


class Runner:
    def __init__(self, args, writer=None, logger=None):
        self.datapath = EasyDict()
        self.datapath.log_dir = os.path.join(args.dump_path, args.exp_name)
        self.datapath.model_path = os.path.join(args.dump_path, 'model')
        self.args = args
        self.writer = writer
        self.logger = logger
        self.scaler = GradScaler()
        self.model_list = []
        self.epoch = -1
        self.data_init()
        self.model_choise()

    def data_init(self):
        self.kgs, self.non_train, self.train_set, self.eval_set, self.test_set, self.test_ill_ = load_data(self.logger,
                                                                                                           self.args)
        self.train_ill = self.train_set.data
        self.eval_left = torch.LongTensor(self.eval_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.eval_set[:, 1].squeeze()).cuda()
        if self.test_set is not None:
            self.test_left = torch.LongTensor()
            self.test_right = torch.LongTensor()

        self.eval_sampler = None
        file_path = os.path.join(self.args.data_path, self.args.data_choice, self.args.data_split)
        self.all_triples, self.node_size, self.rel_size = self.load_triples(file_path + '/', True)
        self.adj = get_adjr(self.node_size, self.all_triples, norm=True).to(self.args.device)

    def load_triples(self, file_path, reverse=True):
        def reverse_triples(triples):
            reversed_triples = np.zeros_like(triples)
            for i in range(len(triples)):
                reversed_triples[i, 0] = triples[i, 2]
                reversed_triples[i, 2] = triples[i, 0]
                if reverse:
                    reverse_triples[i, 1] = triples[i, 1] + rel_size
                else:
                    reverse_triples[i, 1] = triples[i, 1]
            return reverse_triples

        with open(file_path, 'triples_1') as f:
            triples1 = f.readlines()

        with open(file_path, 'triples_2') as f:
            triples2 = f.readlines()

        triples = np.array([line.replace('\n', '').split('\t') for line in triples1 + triples2]).astype(np.int64)
        node_size = max([np.max(triples[:, 0]), np.max(triples[:, 2])]) + 1
        rel_size = np.max(np.max(triples[:, 1])) + 1

        all_triples = np.concatenate([triples, reverse_triples(triples)], axis=0)
        all_triples = np.unique(all_triples, axis=0)

        return all_triples, node_size, rel_size * 2 if reverse else rel_size
