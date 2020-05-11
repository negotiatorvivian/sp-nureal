import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import random

from sp_encoders import Encoder
from aggregators import MeanAggregator
from util import CompactDimacs, Perceptron, update_solution
from base import SATProblem


class SupervisedGraphSage(nn.Module):

    def __init__(self, feature_dim, hidden_dim, enc, name):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc  # sp_encoder
        self.lent = nn.L1Loss()
        self._prediction_layer = Perceptron(feature_dim, hidden_dim, 1)
        self._name = name
        self.weight = nn.Parameter(torch.FloatTensor(feature_dim, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, sat_problem):
        embeds = self.enc(nodes)
        # scores = self.activation(self.weight.mm(embeds).t())
        ''' 线性层计算 score'''
        scores = self._prediction_layer(self.weight.mm(embeds).t())
        mask = torch.ones((sat_problem._variable_num * 2, 1), dtype = torch.float)
        mask[nodes] = 1 - scores
        mask = 1 - mask
        length = int(len(nodes)/2)
        '''更新 sat_problem 的解'''
        variable_prediction = update_solution(mask[:length], sat_problem)
        '''计算 sat_problem 的可满足子句数'''
        sat_problem._post_process_predictions(variable_prediction)
        return scores

    def loss(self, nodes, labels, sat_problem):
        scores = self.forward(nodes, sat_problem)
        return self.lent(scores.squeeze(1), labels)


def load_cora(dimacs_file):
    dimacs = CompactDimacs(dimacs_file)
    return dimacs


def _to_cuda(data, _use_cuda, _device):
    if isinstance(data, list):
        return data
    if data is not None and _use_cuda:
        return data.cuda(_device, non_blocking = True)
    return data


def _module(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def run_cora(data_loader, rep, epoch, model_list, loss_evaluator, device, batch_replication, hidden_dimension,
             feature_dim, train_outer_recurrence_num, use_cuda = True, is_train = True, randomized = True):
    np.random.seed(1)
    random.seed(1)
    '''优化参数列表'''
    optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters())} for model in model_list]
    total_loss = np.zeros(len(model_list), dtype = np.float32)
    total_example_num = 0
    '''# 下标起始位置为1,每次读入为 dalaloader 里的一项'''
    for (j, data) in enumerate(data_loader, 1):
        segment_num = len(data[0])
        for seg in range(segment_num):
            '''将读进来的data放进gpu'''
            (graph_map, batch_variable_map, batch_function_map,
             edge_feature, graph_feat, label, answers, var, func) = [_to_cuda(d[seg], use_cuda, device) for d in data]
            total_example_num += (batch_variable_map.max() + 1)
            sat_problem = SATProblem((graph_map, batch_variable_map, batch_function_map, edge_feature, answers, None),
                                     device, batch_replication)
            '''将所有CNF的答案拼接起来'''
            answers = np.concatenate(answers, axis = 0)
            ''''''
            variable_map = torch.cat(((sat_problem.nodes.to(torch.long) + sum(var)).reshape(1, -1), graph_map[1].to(
                torch.long).reshape(1, -1)), dim = 0)
            '''feat_data 为输入 CNF 的[变量, 子句]矩阵'''
            feat_data = torch.sparse.FloatTensor(variable_map, edge_feature.squeeze(1),
                                                 torch.Size([sum(var) * 2 + 1, sum(func)])).to_dense()
            feat_data = feat_data[np.argwhere(torch.sum(torch.abs(feat_data), 1) > 0)[0]]
            num_nodes_x = feat_data.shape[0]
            num_nodes_y = feat_data.shape[1]
            '''# 编码读入的数据'''
            features = nn.Embedding(num_nodes_x, num_nodes_y)
            '''# 用sp模型初始化的data作为特征值的权重'''
            features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad = False)
            if use_cuda:
                features = features.cuda()

            agg1 = MeanAggregator(features, cuda = use_cuda)
            enc1 = Encoder(features, num_nodes_y, hidden_dimension, sat_problem.adj_lists,
                           sat_problem.node_adj_lists,
                           agg1, gru = True, cuda = use_cuda)
            agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda = use_cuda)
            enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, hidden_dimension,
                           sat_problem.adj_lists, sat_problem.node_adj_lists, agg2, base_model = enc1, gru = False,
                           cuda = use_cuda)
            enc1.num_samples = 15
            enc2.num_samples = 5
            graphsage = SupervisedGraphSage(hidden_dimension, feature_dim, enc2, 'sp-nueral')
            '''优化参数列表增加graphSAGE模型参数'''
            optim_list.append({'params': filter(lambda p: p.requires_grad, graphsage.parameters())})
            optimizer = torch.optim.SGD(optim_list, lr = 0.3, weight_decay = 0.01)
            optimizer.zero_grad()
            loss = torch.zeros(1, device = device)
            nodes = [i for i in range(2 * sat_problem._variable_num)]
            edges = np.concatenate((-1 * sat_problem._meta_data[0], sat_problem._meta_data[0]))
            # edges = torch.tensor(sat_problem._meta_data)[0][torch.abs(sat_problem.nodes).numpy() - 1]
            for i in range(train_outer_recurrence_num):
                loss += graphsage.loss(nodes, Variable(torch.FloatTensor(edges)), sat_problem)
            for (k, model) in enumerate(model_list):
                edges = None
                res = None
                '''初始化变量state, 可选择随机是否随机初始化 其中 batch_replication 表示同一个CNF数据重复次数'''
                state = _module(model).get_init_state(graph_map, randomized, batch_replication)
                '''train_outer_recurrence_num 代表同一组数据重新训练的次数, loss叠加'''
                for i in torch.arange(train_outer_recurrence_num, dtype = torch.int32, device = device):
                    variable_prediction, state = model(init_state = state, sat_problem = sat_problem,
                                                       is_training = True)
                    if is_train:
                        '''计算 sp_aggregator 的loss'''
                        loss += loss_evaluator(variable_prediction = variable_prediction, label = label,
                                               graph_map = sat_problem._graph_map,
                                               batch_variable_map = sat_problem._batch_variable_map,
                                               batch_function_map = sat_problem._batch_function_map,
                                               edge_feature = sat_problem._edge_feature, meta_data = None,
                                               global_step = model._global_step)
                        '''根据训练结果计算CNF预测值 确定某些变量的值'''
                        res = sat_problem._post_process_predictions(variable_prediction)
                        if res is None:
                            break
                        '''取出子句不满足的所有边的集合'''
                        # edges = answers.repeat(batch_replication)[res]
                        # loss_, res = graphsage.loss(res, Variable(torch.FloatTensor(edges)), sat_problem)
                        # loss += loss_
                # if edges is None or res is None:
                #     break
                # for j in range(train_outer_recurrence_num):
                #     '''子句不满足的所有边的集合放入graphSAGE模型训练'''
                #     loss_, res = graphsage.loss(res, Variable(torch.FloatTensor(edges)), sat_problem)
                #     loss += loss_
                #     edges = answers.repeat(batch_replication)[res]
                print('rep: %d, epoch: %d, data segment: %d, loss: %f' % (rep, epoch, seg, loss))
                total_loss[k] += loss.detach().cpu().numpy()
                loss.backward()
                for s in state:
                    del s
            optimizer.step()

        for model in model_list:
            _module(model)._global_step += 1

            del graph_map
            del batch_variable_map
            del batch_function_map
            del graph_feat
            del label
            del edge_feature
    return total_loss / total_example_num
