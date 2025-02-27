import torch
from torch import nn
from torch.nn import functional as F


class ConvTransE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, in_channels=2, out_channels=10, kernel_size=3, drop_out_0=0.2,
                 drop_out_1=0.2, drop_out_2=0.2, device=None):
        super(ConvTransE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.ent_emb = nn.Parameter(torch.zeros(num_ent, emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.ent_emb)
        self.rel_emb = nn.Parameter(torch.zeros(num_rel * 2, emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.rel_emb)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.drop0 = nn.Dropout(drop_out_1)
        self.drop1 = nn.Dropout(drop_out_2)
        self.drop2 = nn.Dropout(drop_out_2)
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(emb_dim)
        self.fc = nn.Linear(in_features=out_channels * emb_dim, out_features=emb_dim)
        self.device = device
        self.loss_e = nn.CrossEntropyLoss()

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

    def forward(self, triples):
        ent_emb = F.tanh(self.ent_emb)
        rel_emb = self.rel_emb
        batch_size = len(triples)
        head = ent_emb[triples[:, 0]].unsqueeze(1)  # [!!!important] [batch_size, 1, embedding_dim]
        rel = rel_emb[triples[:, 1]].unsqueeze(1)  # [!!!important] [batch_size, 1, embedding_dim]
        emb_stack = torch.cat([head, rel], dim=1)  # [!!!important] [batch_size, 2, embedding_dim]

        emb_stack = self.bn0(emb_stack)  # [优化]
        # emb_stack = self.drop0(emb_stack)  # [优化]

        """     常规顺序    """
        x = self.conv1(emb_stack)  # [batch_size, out_channel, embedding_dim]
        x = self.bn1(x)  # [优化]
        x = F.relu(x)  # [优化]
        x = self.drop1(x)  # [优化]

        """     先drop后bn(得到得分函数)    """
        x = x.view(batch_size, -1)  # [batch_size, out_channel * embedding_dim]
        x = self.fc(x)  # [batch_size, embedding_dim]
        x = self.bn2(x)  # [优化]
        x = F.relu(x)  # [优化]
        x = self.drop2(x)  # [优化]

        x = torch.mm(x, ent_emb.transpose(1, 0))  # [batch_size, num_entity]
        return x

    def get_loss(self, origin_triples):
        loss_ent = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        # loss_rel = torch.zeros(1).cuda().to(self.device)  # gpu单卡训练
        inverse_triplets = origin_triples[:, [2, 1, 0]]
        inverse_triplets[:, 1] += self.num_rel
        triples = torch.cat((torch.from_numpy(origin_triples), torch.from_numpy(inverse_triplets)))
        triples = triples.to(self.device)
        score_ent = self.forward(triples=triples)
        # if self.entity_prediction:
        loss_ent += self.loss_e(score_ent, triples[:, 2])
        # if self.relation_prediction:
        #     loss_rel += self.loss_r(score_rel, triples[:, 1])
        # self.loss_r()
        return loss_ent
