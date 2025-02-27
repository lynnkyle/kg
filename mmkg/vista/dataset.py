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

        self.filter_dict = []
        for data_split in [self.train, self.valid, self.test]:
            for triple in data_split:
                h, r, t = triple
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1, h, r)] = []
                self.filter_dict[(-1, h, r)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1)] = []
                self.filter_dict[(h, r, -1)].append(t)

        self.ent_max_vis_len = ent_max_vis_len
        self.rel_max_vis_len = rel_max_vis_len

    def gather_vis_feature(self):
        pass

    def gather_txt_feature(self):
        self.ent2txt = torch.load()


if __name__ == '__main__':
    data = VTKG(data="FB15K237", logger=None)
    print(len(data.train))
    print(len(data.test))
    print(len(data.valid))
