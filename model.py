import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import random

from encoders import Encoder
from aggregators import MeanAggregator
from util import CompactDimacs, Perceptron, update_solution
from solver import SATProblem, SatLossEvaluator


class SupervisedGraphSage(nn.Module):

    def __init__(self, device, feature_dim, hidden_dim, enc, name):
        super(SupervisedGraphSage, self).__init__()
        self._device = device
        self.enc = enc  # sp_encoder
        self.lent = nn.L1Loss()
        # self.lent = nn.CrossEntropyLoss()
        self._prediction_layer = Perceptron(feature_dim, hidden_dim, 1)
        self._name = name
        self.weight = nn.Parameter(torch.FloatTensor(enc.node_adj_lists.size, feature_dim).to(device))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, sat_problem, is_train):
        embeds = self.enc(nodes)
        scores = embeds.mm(self.weight)
        signed_variable_mask_transpose, function_mask = SatLossEvaluator.compute_masks(sat_problem._graph_map, sat_problem._batch_variable_map, sat_problem._batch_function_map, sat_problem._edge_feature, self._device)

        edge_values = scores + (1 - sat_problem._edge_feature) / 2
        if not is_train:
            solution = torch.mm(signed_variable_mask_transpose.to_dense().t(), edge_values)
            # if len(variables) != sat_problem._variable_num:
            #     solution = sat_problem._solution.detach()
            #     solution[nodes] = torch.tensor(variables, dtype = torch.float)
            #     mask = solution.unsqueeze(1)
            # else:
            #     mask = torch.tensor(variables, dtype = torch.float, device = self._device).unsqueeze(1)
            torch.clamp(solution, 0, 1)
            '''计算 sat_problem 的可满足子句数'''
            _, output, certain_vars = sat_problem._post_process_predictions(solution)
            if output:
                sat_problem._active_variables[certain_vars[0] - 1] = 0
                sat_problem._solution[torch.tensor(certain_vars[0] - 1)] = torch.tensor(certain_vars[1],
                                                                                        dtype = torch.float,
                                                                                        device = self._device)
                # mask.numpy()[certain_vars[0]][:, 0] = certain_vars[1]
                '''更新 sat_problem 的解'''
                update_solution(solution, sat_problem)
        edge_values = (edge_values > 0.5).float()
        clause_values = torch.mm(function_mask, edge_values)
        clause_values = (clause_values > 0).float()
        return clause_values

    def loss(self, nodes, sat_problem, is_train):
        clause_values = self.forward(nodes, sat_problem, is_train)
        # return self.lent(scores.squeeze(1), labels)
        return self.lent(clause_values, torch.ones(sat_problem._function_num))


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


def train_batch(solver_base, data_loader, total_loss, rep, epoch, model_list, device, batch_replication, hidden_dimension, feature_dim, train_graph_recurrence_num, train_outer_recurrence_num, use_cuda = True, is_train = True, randomized = True):
    # np.random.seed(1)

    random.seed(1)
    '''优化参数列表'''
    optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters())} for model in model_list]
    total_example_num = 0
    '''# 下标起始位置为1,每次读入为 dalaloader 里的一项'''
    for (j, data) in enumerate(data_loader, 1):
        segment_num = len(data[0])
        print('Train CNF:', j)
        for seg in range(segment_num):
            '''将读进来的data放进gpu'''
            (graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, label, answers, var, func) = [
                _to_cuda(d[seg], use_cuda, device) for d in data]
            total_example_num += (batch_variable_map.max() + 1)
            sat_problem = SATProblem((graph_map, batch_variable_map, batch_function_map, edge_feature, answers, None),
                                     device, batch_replication)
            loss = torch.zeros(1, device = device, requires_grad = False)
            # '''将所有CNF的答案拼接起来, 有解才执行 graphSage 模型'''
            # if len(answers[0].flatten()) > 0:
            # answers = np.concatenate(answers, axis = 0)
            '''展开所有子句的变量(绝对值)'''
            variable_map = torch.cat(((torch.abs(sat_problem.nodes).to(torch.long) - 1).reshape(1, -1),
            graph_map[1].to(torch.long).reshape(1, -1)), dim = 0)
            '''feat_data 为输入 CNF 的[变量, 子句]矩阵'''
            feat_data = torch.sparse.FloatTensor(variable_map, edge_feature.squeeze(1),
                                                 torch.Size([sum(var), sum(func)])).to_dense()
            # feat_data = feat_data[np.argwhere(torch.sum(torch.abs(feat_data), 1) > 0)[0]]
            num_nodes_x = feat_data.shape[0]
            num_nodes_y = feat_data.shape[1]
            '''编码读入的数据'''
            features = nn.Embedding(num_nodes_x, num_nodes_y)
            '''用sp模型初始化的data作为特征值的权重'''
            features.weight = nn.Parameter(feat_data, requires_grad = False)
            if use_cuda:
                features = features.cuda()

            agg1 = MeanAggregator(features, device = device)
            enc1 = Encoder(device, features, num_nodes_y, sat_problem._edge_num, sat_problem.adj_lists,
                           sat_problem.node_adj_lists, agg1, gru = True)
            agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), device = device)
            enc2 = Encoder(device, lambda nodes: enc1(nodes).t(), enc1.embed_dim, sat_problem._edge_num,
                           sat_problem.adj_lists, sat_problem.node_adj_lists, agg2, base_model = enc1, gru = False)
            enc1.num_samples = 15
            enc2.num_samples = 5
            graphsage = SupervisedGraphSage(device, hidden_dimension, feature_dim, enc2, 'sp-nueral')

            '''优化参数列表增加graphSAGE模型参数'''
            optim_list.append({'params': filter(lambda p: p.requires_grad, graphsage.parameters())})
            optimizer = torch.optim.SGD(optim_list, lr = 0.3, weight_decay = 0.01)
            optimizer.zero_grad()
            nodes = [i for i in range(sat_problem._variable_num)]
            # sample_length = int(len(nodes)/train_outer_recurrence_num)
            for i in range(train_graph_recurrence_num):
                loss += graphsage.loss(nodes, sat_problem, i < (train_graph_recurrence_num - 1))
            # else:
            #     # optimizer = torch.optim.SGD(optim_list, lr = 0.3, weight_decay = 0.01)
            #     optimizer = torch.optim.Adam(optim_list, lr = 0.3, weight_decay = 0.01)
            #     optimizer.zero_grad()
            for (k, model) in enumerate(model_list):
                '''初始化变量state, 可选择随机是否随机初始化 其中 batch_replication 表示同一个CNF数据重复次数'''
                state = _module(model).get_init_state(graph_map, randomized, batch_replication)
                '''train_outer_recurrence_num 代表同一组数据重新训练的次数, loss叠加'''
                for i in torch.arange(train_outer_recurrence_num, dtype = torch.int32, device = device):
                    variable_prediction, state = model(init_state = state, sat_problem = sat_problem,
                                                       is_training = True)
                    '''计算 sp_aggregator 的loss'''
                    # loss += model.compute_loss(is_train, variable_prediction, label, sat_problem._graph_map,
                    #                            sat_problem._batch_variable_map, sat_problem._batch_function_map,
                    #                            sat_problem._edge_feature, sat_problem._meta_data)

                    loss += solver_base._compute_loss(_module(model), None, is_train, variable_prediction, label,
                                                      sat_problem)

                    for p in variable_prediction:
                        del p

                for s in state:
                    del s

                print('rep: %d, epoch: %d, data segment: %d, loss: %f' % (rep, epoch, seg, loss))
                total_loss[k] += loss.detach().cpu().numpy()
                loss.backward()

            optimizer.step()

        for model in model_list:
            _module(model)._global_step += 1

            del graph_map
            del batch_variable_map
            del batch_function_map
            del graph_feat
            del label
            del edge_feature
    return total_loss / total_example_num.cpu().numpy()


def test_batch(solver_base, data_loader, errors, model_list, device, batch_replication, use_cuda = True,
        is_train = False, randomized = True):
    np.random.seed(1)
    random.seed(1)
    total_example_num = 0
    '''# 下标起始位置为1,每次读入为 dalaloader 里的一项'''
    for (j, data) in enumerate(data_loader, 1):
        segment_num = len(data[0])
        for seg in range(segment_num):
            '''将读进来的data放进gpu'''
            (graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, label, answers, var, func) = [
                _to_cuda(d[seg], use_cuda, device) for d in data]
            total_example_num += (batch_variable_map.max() + 1)
            sat_problem = SATProblem((graph_map, batch_variable_map, batch_function_map, edge_feature, answers, None), device, batch_replication)
            for (k, model) in enumerate(model_list):
                '''初始化变量state, 可选择随机是否随机初始化 其中 batch_replication 表示同一个CNF数据重复次数'''
                state = _module(model).get_init_state(graph_map, randomized, batch_replication)
                variable_prediction, state = model(init_state = state, sat_problem = sat_problem, is_training = True)
                '''计算 sp_aggregator 的 error'''
                errors[:, k] += solver_base._compute_loss(_module(model), None, is_train, variable_prediction, label, sat_problem)

                for p in variable_prediction:
                    del p

                for s in state:
                    del s

                print('{:s}: segment: {:d}, error={:s}|'.format(_module(model)._name, seg, np.array_str(errors[:, k].flatten())))

            del graph_map
            del batch_variable_map
            del batch_function_map
            del graph_feat
            del label
            del edge_feature
            print('true: %d, false: %d, uncertain: %d' % (
                int(sat_problem.statistics[0]), int(sat_problem.statistics[1]), int(sat_problem.statistics[2])))

    return errors / total_example_num.cpu().numpy()


def predict_batch(data_loader, model_list, device, batch_replication, use_cuda = True, randomized = True):
    np.random.seed(1)
    random.seed(1)
    total_example_num = 0
    '''下标起始位置为1,每次读入为 dalaloader 里的一项'''
    for (j, data) in enumerate(data_loader, 1):
        segment_num = len(data[0])
        for seg in range(segment_num):
            '''将读进来的data放进gpu'''
            (graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, label, answers, var, func) = [
                _to_cuda(d[seg], use_cuda, device) for d in data]
            total_example_num += (batch_variable_map.max() + 1)
            sat_problem = SATProblem((graph_map, batch_variable_map, batch_function_map, edge_feature, answers, None),
                                     device, batch_replication)
            for (k, model) in enumerate(model_list):
                '''初始化变量state, 可选择随机是否随机初始化 其中 batch_replication 表示同一个CNF数据重复次数'''
                state = _module(model).get_init_state(graph_map, randomized, batch_replication)
                '''train_outer_recurrence_num 代表同一组数据重新训练的次数, loss叠加'''
                variable_prediction, state = model(init_state = state, sat_problem = sat_problem, is_training = False)
                '''根据训练结果计算CNF预测值 确定某些变量的值'''
                res, output, _ = sat_problem._post_process_predictions(variable_prediction, False)

                for p in variable_prediction:
                    del p

                for s in state:
                    del s

            del graph_map
            del batch_variable_map
            del batch_function_map
            del graph_feat
            del label
            del edge_feature
