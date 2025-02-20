import torch
from torch import nn


class MMKGDecoder(nn.Module):
    def __init__(self):
        super(MMKGDecoder, self).__init__()
        pass

    def forward(self, input):
        pass


class ConvTransE(nn.Module):
    def __init__(self, emb_dim, in_channels=2, out_channels=10, kernel_size=3):
        super(ConvTransE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.fc = nn.Linear(in_features=out_channels * emb_dim, out_features=emb_dim)

    ''' convtranse的关键代码 '''

    # def forward(self, ent_emb, rel_emb, triples):
    #     batch_size = len(triples)
    #     emb_ent = ent_emb[triples[:, 0]].unsequeeze(1)  # [!!!important] [batch_size, 1, embedding_dim]
    #     emb_rel = rel_emb[triples[:, 1]].unsequeeze(1)  # [!!!important] [batch_size, 1, embedding_dim]
    #     emb_stack = torch.cat([emb_ent, emb_rel], dim=1)  # [!!!important] [batch_size, 2, embedding_dim]
    #     x = self.conv1(emb_stack)  # [batch_size, out_channel, embedding_dim]
    #     x = x.view(batch_size, -1)  # [batch_size, out_channel * embedding_dim]
    #     x = self.fc(x)  # [batch_size, embedding_dim]
    #     x = torch.mm(x, emb_ent.transpose(1, 0))  # [batch_size, num_entity]
    #     return x

    def forward(self, ent_emb, rel_emb, triples):
        batch_size = len(triples)
        emb_ent = ent_emb[triples[:, 0]].unsequeeze(1)  # [!!!important] [batch_size, 1, embedding_dim]
        emb_rel = rel_emb[triples[:, 1]].unsequeeze(1)  # [!!!important] [batch_size, 1, embedding_dim]
        emb_stack = torch.cat([emb_ent, emb_rel], dim=1)  # [!!!important] [batch_size, 2, embedding_dim]
        x = self.conv1(emb_stack)  # [batch_size, out_channel, embedding_dim]
        x = x.view(batch_size, -1)  # [batch_size, out_channel * embedding_dim]
        x = self.fc(x)  # [batch_size, embedding_dim]
        x = torch.mm(x, emb_ent.transpose(1, 0))  # [batch_size, num_entity]
        return x
