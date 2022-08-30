from torch.optim import Adam
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv


"""模型出自 CSGNN: Contrastive Self-Supervised Graph Neural Network for Molecular Interaction Prediction"""
# Discriminator Layer
class Discriminator(nn.Module):

    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

# AvgReadout Layer
class AvgReadout(nn.Module):

    def __int__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

# MLP Layer
class MLP(nn.Module):

    def __int__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.Linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.Linear2 = nn.Linear(2 * out_channels, out_channels)


    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = self.Linear2(x)
        return x

# CSGNN Layer
class CSGNN(nn.Module):
    def __init__(self, aggregator, feature, hidden1, hidden2, decoder1, dropout):
        super(CSGNN, self).__init__()

        if aggregator == 'GIN':
            self.mlp_o1 = MLP(feature, hidden1)
            self.mlp_o2 = MLP(hidden1 * 2, hidden2)
            self.mlp_s1 = MLP(feature, hidden1)
            self.mlp_s2 = MLP(hidden1 * 2, hidden2)

            self.encoder_o1 = GINConv(self.mlp_o1, train_eps=True).jittable()
            self.encoder_o2 = GINConv(self.mlp_o2, train_eps=True).jittable()
            self.encoder_s1 = GINConv(self.mlp_s1, train_eps=True).jittable()
            self.encoder_s2 = GINConv(self.mlp_s2, train_eps=True).jittable()

        elif aggregator == 'GCN':
            self.encoder_o1 = GCNConv(feature, hidden1)
            self.encoder_o2 = GCNConv(hidden1 * 2, hidden2)
            self.encoder_s1 = GCNConv(feature, hidden1)
            self.encoder_s2 = GCNConv(hidden1 * 2, hidden2)

        self.decoder1 = nn.Linear(hidden2 * 2 * 4, decoder1)
        self.decoder2 = nn.Linear(decoder1, 1)

        self.disc = Discriminator(hidden2 * 2)

        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def forward(self, data_o, data_s, data_a, idx):
        x_o, adj = data_o.x, data_o.edge_index
        adj2 = data_s.edge_index
        x_a = data_a.x

        x1_o = F.relu(self.encoder_o1(x_o, adj))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x1_s = F.relu(self.encoder_s1(x_o, adj2))
        x1_s = F.dropout(x1_s, self.dropout, training=self.training)

        x1_os = torch.cat((x1_o, x1_s), dim=1)

        x2_o = self.encoder_o2(x1_os, adj)
        x2_s = self.encoder_s2(x1_os, adj2)

        x2_os = torch.cat((x2_o, x2_s), dim=1)  # 拼接一二阶聚合向量之后的节点表示向量

        x1_o_a = F.relu(self.encoder_o1(x_a, adj))
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)
        x1_s_a = F.relu(self.encoder_s1(x_a, adj2))
        x1_s_a = F.dropout(x1_s_a, self.dropout, training=self.training)

        x1_os_a = torch.cat((x1_o_a, x1_s_a), dim=1)

        x2_o_a = self.encoder_o2(x1_os_a, adj)
        x2_s_a = self.encoder_s2(x1_os_a, adj2)

        x2_os_a = torch.cat((x2_o_a, x2_s_a), dim=1)  # 拼接一二阶聚合向量之后的假节点输出向量

        # graph representation
        h_os = self.read(x2_os)
        h_os = self.sigm(h_os)

        h_os_a = self.read(x2_os_a)
        h_os_a = self.sigm(h_os_a)

        # adversarial learning
        ret_os = self.disc(h_os, x2_os, x2_os_a)
        ret_os_a = self.disc(h_os_a, x2_os_a, x2_os)

        entity1 = x2_os[idx[0]]
        entity2 = x2_os[idx[1]]

        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = torch.cat((entity1, entity2), dim=1)

        feature = torch.cat((add, product, concatenate), dim=1)

        # decoder
        log = F.relu(self.decoder1(feature))
        log = self.decoder2(log)

        return log, ret_os, ret_os_a, x2_os


# 创建 CSGNN 模型
def Create_model(args):
    model = CSGNN(aggregator=args.aggregator, feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2,
                  decoder1=args.decoder1, dropout=args.dropout)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer
