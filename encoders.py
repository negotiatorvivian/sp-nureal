import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, device, features, feature_dim, embed_dim, adj_lists, node_adj_lists, aggregator, num_sample = 10,
                 base_model = None, gru = False, gru_hidden_dim = 300):
        super(Encoder, self).__init__()
        self._device = device
        '''用sp模型初始化的data作为特征值的权重'''
        self.features = features
        self.feat_dim = feature_dim
        '''与节点n相关的节点'''
        self.adj_lists = adj_lists
        self.node_adj_lists = np.array([l.numpy() for l in node_adj_lists])
        # self.node_adj_lists = np.array(node_adj_lists)
        self.aggregator = aggregator
        '''训练节点n取与之相关节点的个数'''
        self.num_sample = num_sample
        if base_model is not None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.aggregator.device = device
        self.gru = nn.GRUCell(self.feat_dim, gru_hidden_dim, bias = True).to(self._device) if gru else False
        # self.weight = nn.Parameter(
        #         torch.FloatTensor(embed_dim, gru_hidden_dim if gru else 2 * self.feat_dim))
        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, gru_hidden_dim if gru else 2 * self.feat_dim).to(device))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """获取与n相关的节点m的相关节点编码"""
        neigh_feats = self.aggregator.forward(nodes, [list(self.adj_lists[int(node)].cpu().numpy()) for node in nodes], self.num_sample)
        if not self.gru:
            '''用sp模型初始化的data作为特征值的权重对相关节点的编码'''
            self_feats = self.features(torch.LongTensor(nodes).to(self._device))
            combined = torch.cat([self_feats, neigh_feats], dim = 1)
        else:
            combined = self.gru(neigh_feats)
        combined = F.relu(self.weight.mm(combined.t()))
        return combined
