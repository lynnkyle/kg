import os.path

import torch
from torch.utils.data import Dataset


class VTKG(Dataset):
    def __init__(self, data, logger, ent_max_vis_len=-1, rel_max_vis_len=-1):
        self.data = data
        self.logger = logger
        self.dir = f"../data/{data}/"
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []
        with open(self.dir + "entities.txt", 'r') as f:
            for idx, line in enumerate(f.readlines()):
                self.ent2id[line.strip()] = idx
                self.id2ent.append(line.strip())
        self.num_ent = len(self.ent2id)

        with open(self.dir + "relations.txt", 'r') as f:
            for idx, line in enumerate(f.readlines()):
                self.rel2id[line.strip()] = idx
                self.id2rel.append(line.strip())
        self.num_rel = len(self.rel2id)

        self.train = []
        with open(self.dir + "train.txt", 'r') as f:
            for idx, line in enumerate(f.readlines()):
                h, r, t = line.strip().split("\t")
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.test = []
        with open(self.dir + "test.txt", 'r') as f:
            for idx, line in enumerate(f.readlines()):
                h, r, t = line.strip().split("\t")
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.valid = []
        with open(self.dir + "valid.txt", 'r') as f:
            for idx, line in enumerate(f.readlines()):
                h, r, t = line.strip().split("\t")
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.filter_dict = {}
        for data_split in [self.train, self.valid, self.test]:
            for triple in data_split:
                h, r, t = triple
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1, r, t)] = []
                self.filter_dict[(-1, r, t)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1)] = []
                self.filter_dict[(h, r, -1)].append(t)

        self.ent_max_vis_len = ent_max_vis_len
        self.rel_max_vis_len = rel_max_vis_len
        self.gather_vis_feature()
        self.gather_txt_feature()

    def gather_vis_feature(self):
        self.ent2vis = torch.load(self.dir + 'visual_features_ent_sorted.pt')
        # self.rel2vis = torch.load(self.dir + 'visual_features_rel_sorted.pt')
        self.vis_feat_size = len(self.ent2vis[list(self.ent2vis.keys())[0]][0])

        total_num = 0
        if self.ent_max_vis_len != -1:
            for ent_name in self.ent2vis:
                num_feats = len(self.ent2vis[ent_name])
                total_num += num_feats
                self.ent2vis[ent_name] = self.ent2vis[ent_name][:self.ent_max_vis_len]
            for rel_name in self.rel2vis:
                self.rel2vis[rel_name] = self.rel2vis[rel_name][:self.rel_max_vis_len]
        else:
            for ent_name in self.ent2vis:
                num_feats = len(self.ent2vis[ent_name])
                total_num += num_feats
                if self.ent_max_vis_len < len(self.rel2vis[ent_name]):
                    self.ent_max_vis_len = len(self.rel2vis[ent_name])
            self.ent_max_vis_len = max(self.ent_max_vis_len, 0)
            for rel_name in self.rel2vis:
                if self.rel_max_vis_len < len(self.rel2vis[rel_name]):
                    self.rel_max_vis_len = len(self.rel2vis[rel_name])
            self.ent_max_vis_len = max(self.rel_max_vis_len, 0)

        self.ent_vis_mask = torch.full((self.num_ent, self.ent_max_vis_len),)
        self.ent_vis_matrix = torch.zeros()
        self.rel_vis_mask = torch.full()
        self.rel_vis_matrix = torch.zeros()

    def gather_txt_feature(self):
        self.ent2txt = torch.load(self.dir + 'textual_features_ent.pt')
        self.rel2txt = torch.load(self.dir + 'textual_features_rel.pt')
        self.txt_feat_dim = len(self.ent2txt[list(self.ent2txt.keys())[0]])
        self.ent_txt_matrix = torch.zeros((self.num_ent, self.txt_feat_dim)).cuda()
        self.rel_txt_matrix = torch.zeros((self.num_rel, self.txt_feat_dim)).cuda()
        for ent_name in self.ent2id:
            self.ent_txt_matrix[self.ent2id[ent_name]] = self.ent2txt[ent_name]
        for rel_name in self.rel2id:
            self.rel_txt_matrix[self.rel2id[rel_name]] = self.rel2txt[rel_name]


if __name__ == '__main__':
    data = VTKG(data="FB15K237", logger=None)
