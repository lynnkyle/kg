import torch
from torch import nn


class MMKGDecoder(nn.Module):
    def __init__(self):
        super(MMKGDecoder, self).__init__()
        pass

    def forward(self, input):
        pass


class ConvTransE(nn.Module):
    def __init__(self):
        super(ConvTransE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3)

    def forward(self, ent_emb, rel_emb, triples):
        emb_ent = ent_emb[triples[:, 0]].unsequeeze(1)
        emb_rel = rel_emb[triples[:, 1]].unsequeeze(1)
        emb_stack = torch.cat([emb_ent, emb_rel], dim=1)
        x = self.conv1(emb_stack)


if __name__ == '__main__':
