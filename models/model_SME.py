import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearLayer, self).__init__()
        self.lmat = nn.Parameter(nn.init.xavier_normal_(torch.empty((in_dim, out_dim))))
        self.rmat = nn.Parameter(nn.init.xavier_normal_(torch.empty((in_dim, out_dim))))
        self.bias = nn.Parameter(nn.init.xavier_normal_(torch.empty((1, out_dim))))

    # linput/rinput: .shape: (batch_size, in_dim)
    def forward(self, linput, rinput):
        # out: .shape: (batch_size, out_dim)
        return torch.mm(linput, self.lmat) + torch.mm(rinput, self.rmat) + self.bias


class SME(nn.Module):
    def __init__(self, n_ent, n_rel, depth, hidden, margin, reg):
        super(SME, self).__init__()

        self.margin = margin
        self.depth = depth
        self.hidden = hidden
        self.reg = reg
        self.ent_embedding = nn.Embedding(n_ent, depth)
        self.rel_embedding = nn.Embedding(n_rel, depth)

        self.lll = LinearLayer(depth, hidden)
        self.rll = LinearLayer(depth, hidden)

        #self.all_params = [self.ent_embedding.weight, self.rel_embedding.weight, ]

    def initialize(self):
        nn.init.xavier_normal_(self.ent_embedding)
        nn.init.xavier_normal_(self.rel_embedding)
        # self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, dim=1)

    def get_score(self, heads, tails, rels):
        # shape: (batch_size, depth)
        heads, tails, rels = self.ent_embedding(heads), self.ent_embedding(tails), self.rel_embedding(rels)
        l_out = self.lll(heads, rels)
        r_out = self.rll(tails, rels)
        # return shape: (batch_size,)
        return -torch.sum(l_out * r_out, axis=1)

    def forward(self, pos_x, neg_x):
        self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)
        # self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, dim=1)
        # shape: (batch_size,)
        pos_heads, pos_tails, pos_rels = pos_x[:, 0], pos_x[:, 1], pos_x[:, 2]
        neg_heads, neg_tails, neg_rels = neg_x[:, 0], neg_x[:, 1], neg_x[:, 2]
        pos_score = self.get_score(pos_heads, pos_tails, pos_rels)
        neg_score = self.get_score(neg_heads, neg_tails, neg_rels)
        return torch.max((self.margin + pos_score - neg_score), torch.tensor([0.]).to(device)).mean()

