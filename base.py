import multiprocessing
import torch
import torch.nn as nn
import os
import time
import argparse
import traceback
import numpy as np
from collections import defaultdict
from itertools import combinations

import model as SPModel
from aggregators import SurveyAggregator, SurveyNeuralPredictor
import util, solver


def _module(model):
    """如果可以并行化, 则并行处理"""
    return model.module if isinstance(model, nn.DataParallel) else model


class SATProblem(object):

    def __init__(self, data_batch, device, batch_replication = 1):
        self._device = device
        self.batch_replication = batch_replication
        self.setup_problem(data_batch, batch_replication)
        self._edge_mask = None
        self._is_sat = 0.5 * torch.ones(self._batch_size, device = self._device)
        self._cnf_evaluator = solver.SatCNFEvaluator(device)

    def setup_problem(self, data_batch, batch_replication):
        """
        根据读入文件建立一个sat_problem类
        """
        '''batch_replication > 1, 将数据重复batch_replication次数'''
        if batch_replication > 1:
            self._replication_mask_tuple = self._compute_batch_replication_map(data_batch[1], batch_replication)
            self._graph_map, self._batch_variable_map, self._batch_function_map, self._edge_feature, self._meta_data, _ \
                = self._replicate_batch(data_batch, batch_replication)
        else:
            self._graph_map, self._batch_variable_map, self._batch_function_map, self._edge_feature, self._meta_data, _ \
                = data_batch

        self._variable_num = self._batch_variable_map.size()[0]
        self._function_num = self._batch_function_map.size()[0]
        self._edge_num = self._graph_map.size()[1]

        '''# 以下为关于变量和子句的编码, 数据类型为稀疏矩阵'''
        self._vf_mask_tuple = self._compute_variable_function_map(self._graph_map, self._batch_variable_map,
                                                                  self._batch_function_map, self._edge_feature)
        self._batch_mask_tuple = self._compute_batch_map(self._batch_variable_map, self._batch_function_map)
        self._graph_mask_tuple = self._compute_graph_mask(self._graph_map, self._batch_variable_map,
                                                          self._batch_function_map,
                                                         degree = True)
        '''# 所有edge_feature为正的子句'''
        self._pos_mask_tuple = self._compute_graph_mask(self._graph_map, self._batch_variable_map,
                                                        self._batch_function_map,
                                                        (self._edge_feature == 1).squeeze(1).float())
        '''# 所有edge_feature为负的子句'''
        self._neg_mask_tuple = self._compute_graph_mask(self._graph_map, self._batch_variable_map,
                                                        self._batch_function_map,
                                                        (self._edge_feature == -1).squeeze(1).float())
        self._signed_mask_tuple = self._compute_graph_mask(self._graph_map, self._batch_variable_map,
                                                           self._batch_function_map, self._edge_feature.squeeze(1))

        self._active_variables = torch.ones(self._variable_num, 1, device = self._device)
        self._active_functions = torch.ones(self._function_num, 1, device = self._device)
        self._solution = 0.5 * torch.ones(self._variable_num, device = self._device)

        self._batch_size = (self._batch_variable_map.max() + 1).long().item()
        '''# 与某个节点n有关的所有节点->与变量n同时出现在一个子句中'''
        self.adj_lists, self.node_adj_lists = self._compute_adj_list()

    def _replicate_batch(self, data_batch, batch_replication):
        """batch_replication 将一组数据重复多次"""

        graph_map, batch_variable_map, batch_function_map, edge_feature, meta_data, label = data_batch
        edge_num = graph_map.size()[1]
        batch_size = (batch_variable_map.max() + 1).long().item()
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]

        ind = torch.arange(batch_replication, dtype = torch.int32, device = self._device).unsqueeze(1). \
            repeat(1, edge_num).view(1, -1)
        graph_map = graph_map.repeat(1, batch_replication) + ind.repeat(2, 1) * torch.tensor(
            [[variable_num], [function_num]], dtype = torch.int32, device = self._device)

        ind = torch.arange(batch_replication, dtype = torch.int32, device = self._device).unsqueeze(1).\
            repeat(1, variable_num).view(1, -1)
        batch_variable_map = batch_variable_map.repeat(batch_replication) + ind * batch_size
        ind = torch.arange(batch_replication, dtype = torch.int32, device = self._device).unsqueeze(1).\
            repeat(1, function_num).view(1, -1)
        batch_function_map = batch_function_map.repeat(batch_replication) + ind * batch_size
        edge_feature = edge_feature.repeat(batch_replication, 1)

        if meta_data is not None:
            meta_data = torch.tensor(np.concatenate(meta_data)).repeat(batch_replication, 1)

        if label is not None:
            label = label.repeat(batch_replication, 1)

        return graph_map, batch_variable_map.squeeze(0), batch_function_map.squeeze(0), edge_feature, meta_data, label

    def _compute_batch_replication_map(self, batch_variable_map, batch_replication):
        batch_size = (batch_variable_map.max() + 1).long().item()
        x_ind = torch.arange(batch_size * batch_replication, dtype = torch.int64, device = self._device)
        y_ind = torch.arange(batch_size, dtype = torch.int64, device = self._device).repeat(batch_replication)
        ind = torch.stack([x_ind, y_ind])
        all_ones = torch.ones(batch_size * batch_replication, device = self._device)

        if self._device.type == 'cuda':
            mask = torch.cuda.sparse.FloatTensor(ind, all_ones,
                                                 torch.Size([batch_size * batch_replication, batch_size]),
                                                 device = self._device)
        else:
            mask = torch.sparse.FloatTensor(ind, all_ones,
                                            torch.Size([batch_size * batch_replication, batch_size]),
                                            device = self._device)

        mask_transpose = mask.transpose(0, 1)
        return (mask, mask_transpose)

    def _compute_variable_function_map(self, graph_map, batch_variable_map, batch_function_map, edge_feature):
        """_variable_function_map 为子句和变量关联的矩阵"""
        edge_num = graph_map.size()[1]
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]
        all_ones = torch.ones(edge_num, device = self._device)

        if self._device.type == 'cuda':
            mask = torch.cuda.sparse.FloatTensor(graph_map.long(), all_ones,
                                                 torch.Size([variable_num, function_num]), device = self._device)
            signed_mask = torch.cuda.sparse.FloatTensor(graph_map.long(), edge_feature.squeeze(1),
                                                        torch.Size([variable_num, function_num]), device = self._device)
        else:
            mask = torch.sparse.FloatTensor(graph_map.long(), all_ones,
                                            torch.Size([variable_num, function_num]), device = self._device)
            signed_mask = torch.sparse.FloatTensor(graph_map.long(), edge_feature.squeeze(1),
                                                   torch.Size([variable_num, function_num]), device = self._device)

        mask_transpose = mask.transpose(0, 1)
        signed_mask_transpose = signed_mask.transpose(0, 1)

        return mask, mask_transpose, signed_mask, signed_mask_transpose

    def _compute_batch_map(self, batch_variable_map, batch_function_map):
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]
        variable_all_ones = torch.ones(variable_num, device = self._device)
        function_all_ones = torch.ones(function_num, device = self._device)
        variable_range = torch.arange(variable_num, dtype = torch.int64, device = self._device)
        function_range = torch.arange(function_num, dtype = torch.int64, device = self._device)
        batch_size = (batch_variable_map.max() + 1).long().item()

        variable_sparse_ind = torch.stack([variable_range, batch_variable_map.long()])
        function_sparse_ind = torch.stack([function_range, batch_function_map.long()])

        if self._device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, variable_all_ones,
                                                          torch.Size([variable_num, batch_size]), device = self._device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, function_all_ones,
                                                          torch.Size([function_num, batch_size]), device = self._device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, variable_all_ones,
                                                     torch.Size([variable_num, batch_size]), device = self._device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, function_all_ones,
                                                     torch.Size([function_num, batch_size]), device = self._device)

        variable_mask_transpose = variable_mask.transpose(0, 1)
        function_mask_transpose = function_mask.transpose(0, 1)

        return variable_mask, variable_mask_transpose, function_mask, function_mask_transpose

    def _compute_graph_mask(self, graph_map, batch_variable_map, batch_function_map, edge_values = None,
                            degree = False):
        """_graph_mask表示变量-子句-edge的关系"""
        edge_num = graph_map.size()[1]
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]
        neg_prop_flag = False

        if edge_values is None:
            edge_values = torch.ones(edge_num, device = self._device)
        else:
            neg_prop_flag = True

        edge_num_range = torch.arange(edge_num, dtype = torch.int64, device = self._device)

        variable_sparse_ind = torch.stack([graph_map[0, :].long(), edge_num_range])
        function_sparse_ind = torch.stack([graph_map[1, :].long(), edge_num_range])

        if self._device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, edge_values,
                                                          torch.Size([variable_num, edge_num]), device = self._device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, edge_values,
                                                          torch.Size([function_num, edge_num]), device = self._device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, edge_values,
                                                     torch.Size([variable_num, edge_num]), device = self._device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, edge_values,
                                                     torch.Size([function_num, edge_num]), device = self._device)
        if degree:
            self.degree = torch.sum(variable_mask.to_dense(), dim = 1)
        if neg_prop_flag:
            self.neg_prop = torch.sum(variable_mask.to_dense(), dim = 1)

        variable_mask_transpose = variable_mask.transpose(0, 1)
        function_mask_transpose = function_mask.transpose(0, 1)

        return variable_mask, variable_mask_transpose, function_mask, function_mask_transpose

    def _peel(self):

        vf_map, vf_map_transpose, signed_vf_map, _ = self._vf_mask_tuple
        '''子句中有几个变量'''
        variable_degree = torch.mm(vf_map, self._active_functions)
        signed_variable_degree = torch.mm(signed_vf_map, self._active_functions)

        while True:
            single_variables = (variable_degree == signed_variable_degree.abs()).float() * self._active_variables

            if torch.sum(single_variables) <= 0:
                break

            single_functions = (torch.mm(vf_map_transpose, single_variables) > 0).float() * self._active_functions
            degree_delta = torch.mm(vf_map, single_functions) * self._active_variables
            signed_degree_delta = torch.mm(signed_vf_map, single_functions) * self._active_variables
            self._solution[single_variables[:, 0] == 1] = (signed_variable_degree[
                                                               single_variables[:, 0] == 1, 0].sign() + 1) / 2.0

            variable_degree -= degree_delta
            signed_variable_degree -= signed_degree_delta

            self._active_variables[single_variables[:, 0] == 1, 0] = 0
            self._active_functions[single_functions[:, 0] == 1, 0] = 0

    def _set_variable_core(self, assignment):
        """固定某些可以确定的值"""

        _, vf_map_transpose, _, signed_vf_map_transpose = self._vf_mask_tuple

        assignment *= self._active_variables

        '''# 计算子句节点的输入(无符号)'''
        input_num = torch.mm(vf_map_transpose, assignment.abs())

        '''# 计算子句节点的输入(有符号)'''
        function_eval = torch.mm(signed_vf_map_transpose, assignment)

        deactivated_functions = (function_eval > -input_num).float() * self._active_functions

        self._active_variables[assignment[:, 0].abs() == 1, 0] = 0
        self._active_functions[deactivated_functions[:, 0] == 1, 0] = 0

        '''更新解'''
        self._solution[assignment[:, 0].abs() == 1] = (assignment[assignment[:, 0].abs() == 1, 0] + 1) / 2.0

    def _propagate_single_clauses(self):
        "Implements unit clause propagation algorithm."

        vf_map, vf_map_transpose, signed_vf_map, _ = self._vf_mask_tuple
        b_variable_mask, b_variable_mask_transpose, b_function_mask, _ = self._batch_mask_tuple

        while True:
            function_degree = torch.mm(vf_map_transpose, self._active_variables)
            single_functions = (function_degree == 1).float() * self._active_functions

            if torch.sum(single_functions) <= 0:
                break
            input_num = torch.mm(vf_map, single_functions)

            '''有符号的变量值'''
            variable_eval = torch.mm(signed_vf_map, single_functions)

            '''判断是否有冲突的子句或变量'''
            conflict_variables = (variable_eval.abs() != input_num).float() * self._active_variables
            if torch.sum(conflict_variables) > 0:
                '''是否有不可解的子句'''
                unsat_examples = torch.mm(b_variable_mask_transpose, conflict_variables)
                self._is_sat[unsat_examples[:, 0] >= 1] = 0

                '''将不可解的子句关联到不满足的集合中'''
                unsat_functions = torch.mm(b_function_mask, unsat_examples) * self._active_functions
                self._active_functions[unsat_functions[:, 0] == 1, 0] = 0

                '''根据不满足的子句集合 -> 不满足的变量集合'''
                unsat_variables = torch.mm(b_variable_mask, unsat_examples) * self._active_variables
                self._active_variables[unsat_variables[:, 0] == 1, 0] = 0

            '''计算活跃变量的值'''
            assigned_variables = (variable_eval.abs() == input_num).float() * self._active_variables

            '''计算所有变量赋值'''
            assignment = torch.sign(variable_eval) * assigned_variables

            '''确定的变量 -> 确定的子句'''
            self._active_functions[single_functions[:, 0] == 1, 0] = 0

            '''根据 assignment 确定相关的变量的值'''
            self._set_variable_core(assignment)

    def set_variables(self, assignment):
        """将某些变量值确定"""
        self._set_variable_core(assignment)
        self.simplify()

    def simplify(self):
        """化简 CNF """
        self._propagate_single_clauses()
        self._peel()

    def _compute_adj_list(self):
        """计算与变量 n 同时出现在某一个子句中的变量集合"""
        adj_lists = defaultdict(set)
        node_adj_lists = []
        self.nodes = ((self._signed_mask_tuple[0]._indices()[0].to(torch.float) + 1) * self._edge_feature.squeeze(1))\
            .to(torch.long)
        # if self._device.type == 'cuda':
        #     graph_map = self._graph_map.detach().cpu().numpy()
        #     self.nodes = self.nodes.to(self._device)
        for i in range(1, self._variable_num + 1):
            for j in [i, -i]:
                indices = self._graph_map[1][np.argwhere(self.nodes == j)][0].to(torch.long)
                edge_indices = np.argwhere(self._signed_mask_tuple[2].to_dense()[indices] != 0)[1]
                # relations = np.argwhere(self._vf_mask_tuple[1].to_dense()[indices] > 0)[1]

                relations = set([i + self._variable_num if i < 0 else i + self._variable_num - 1 for i in
                                 self.nodes[edge_indices].numpy()])
                index = j + self._variable_num if j < 0 else j + self._variable_num - 1
                node_adj_lists.extend(list(relations - set([index])))
                if len(relations) < 2:
                    continue
                adj_lists[index] = relations - set([index])
        return adj_lists, node_adj_lists

    def _post_process_predictions(self, prediction, answers = None):

        res, activations = self._cnf_evaluator(prediction, self._graph_map, self._batch_variable_map,
                                                self._batch_function_map, self._edge_feature,
                                                self._vf_mask_tuple[1], self._graph_mask_tuple[3],
                                                self._active_variables, self._active_functions)
        trained_vars, functions, variables = activations
        self._active_functions[functions] = 0
        self._active_variables[variables] = 0
        if res is None:
            return None
        output, unsat_clause_num, graph_map, clause_values = [a.detach().cpu().numpy() for a in res]
        print('unsat_clause_num:', unsat_clause_num)
        '''若res不为空, trained_vars表示所有不满足子句中出现过的变量'''
        return trained_vars


class PropagatorDecimatorSolverBase(nn.Module):
    """SP-NUERAL Solver 的基类"""

    def __init__(self, device, name, feature_dim, hidden_dimension,
                 agg_hidden_dimension = 100, func_hidden_dimension = 100, agg_func_dimension = 50,
                 classifier_dimension = 50, local_search_iterations = 1000, epsilon = 0.05, pure_sp = True):

        super(PropagatorDecimatorSolverBase, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()
        self._edge_dimension = 1
        self._meta_dimension = 1
        self._predict_dimension = 1
        self._feature_dim = feature_dim
        self._output_dimension = hidden_dimension + self._edge_dimension + self._meta_dimension
        '''用于变量值的预测'''
        self._variable_classifier = util.Perceptron(hidden_dimension, classifier_dimension, self._predict_dimension)
        self._propagator = SurveyAggregator(self._device, hidden_dimension, feature_dim, self._edge_dimension,
                                            self._meta_dimension, True)

        self._predictor = SurveyNeuralPredictor(device, hidden_dimension, self._predict_dimension, self._edge_dimension,
                                                self._meta_dimension, agg_hidden_dimension, func_hidden_dimension,
                                                agg_func_dimension, self._variable_classifier)

        self._module_list.append(self._propagator)
        self._module_list.append(self._predictor)

        self._global_step = nn.Parameter(torch.tensor([0], dtype=torch.float, device=self._device), requires_grad=False)
        self._name = name
        self._local_search_iterations = local_search_iterations
        self._epsilon = epsilon
        '''是否使用神经网络的 decimate 方法'''
        self._pure_sp = pure_sp

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, init_state, sat_problem, is_training = True, iteration_num = 1, check_termination = None,
                simplify = True, batch_replication = 1):

        init_propagator_state, init_decimator_state = init_state
        batch_replication = 1 if is_training else batch_replication
        '''化简 CNF (是否有冲突或关联的变量)'''
        if simplify and not is_training:
            sat_problem.simplify()

        if self._propagator is not None and self._predictor is not None:
            propagator_state, decimator_state = self._forward_core(init_propagator_state, init_decimator_state,
                                                                   sat_problem, iteration_num, is_training,
                                                                   check_termination)
        else:
            decimator_state = None
            propagator_state = None

        prediction = self._predictor(decimator_state, sat_problem, True)

        # Post-processing local search
        if not is_training:
            prediction = self._local_search(prediction, sat_problem, batch_replication)

        prediction = util.update_solution(prediction, sat_problem)

        if batch_replication > 1:
            prediction, propagator_state, decimator_state = self._deduplicate(prediction, propagator_state,
                                                                              decimator_state, sat_problem)

        return (prediction, (propagator_state, decimator_state))

    def _forward_core(self, init_propagator_state, init_decimator_state, sat_problem, iteration_num, is_training,
                      check_termination):

        propagator_state = init_propagator_state
        decimator_state = init_decimator_state
        '''当找到 CNF 的解时, check_termination 不为 None'''
        if check_termination is None:
            active_mask = None
        else:
            active_mask = torch.ones(sat_problem._batch_size, 1, dtype = torch.uint8, device = self._device)

        for _ in torch.arange(iteration_num, dtype = torch.int32, device = self._device):
            '''调用 survey_propagator'''
            propagator_state, decimator_state = self._propagator(propagator_state, decimator_state, sat_problem,
                                                                 is_training, pure_sp = self._pure_sp)

            sat_problem._edge_mask = torch.mm(sat_problem._graph_mask_tuple[1], sat_problem._active_variables) * \
                                     torch.mm(sat_problem._graph_mask_tuple[3], sat_problem._active_functions)

            if sat_problem._edge_mask.sum() < sat_problem._edge_num:
                decimator_state += (sat_problem._edge_mask,)

            if check_termination is not None:
                '''求 prediction, prediction 是所有 variable 的预测值'''
                prediction = self._predictor(decimator_state, sat_problem)
                '''更新 solution'''
                prediction = util.update_solution(prediction, sat_problem)

                check_termination(active_mask, prediction, sat_problem)
                num_active = active_mask.sum()

                if num_active <= 0:
                    break

        return propagator_state, decimator_state

    def _deduplicate(self, prediction, propagator_state, decimator_state, sat_problem):
        """通过为每个问题实例查找具有最低loss的副本，对当前批次进行重复数据的删除（以抵消批次复制）"""

        if sat_problem._batch_replication <= 1 or sat_problem._replication_mask_tuple is None:
            return None, None, None

        assignment = 2 * prediction[0] - 1.0
        energy, _ = self._compute_energy(assignment, sat_problem)
        max_ind = util.sparse_argmax(-energy.squeeze(1), sat_problem._replication_mask_tuple[0], device = self._device)

        batch_flag = torch.zeros(sat_problem._batch_size, 1, device = self._device)
        batch_flag[max_ind, 0] = 1

        flag = torch.mm(sat_problem._batch_mask_tuple[0], batch_flag)
        variable_prediction = (flag * prediction[0]).view(sat_problem._batch_replication, -1).sum(dim = 0).unsqueeze(1)

        flag = torch.mm(sat_problem._graph_mask_tuple[1], flag)
        new_propagator_state = ()
        for x in propagator_state:
            new_propagator_state += ((flag * x).view(sat_problem._batch_replication, sat_problem._edge_num /
                                                     sat_problem._batch_replication, -1).sum(dim = 0),)

        new_decimator_state = ()
        for x in decimator_state:
            new_decimator_state += ((flag * x).view(sat_problem._batch_replication, sat_problem._edge_num /
                                                    sat_problem._batch_replication, -1).sum(dim = 0),)

        function_prediction = None
        if prediction[1] is not None:
            flag = torch.mm(sat_problem._batch_mask_tuple[2], batch_flag)
            function_prediction = (flag * prediction[1]).view(sat_problem._batch_replication, -1).sum(dim = 0)\
                .unsqueeze(1)

        return (variable_prediction, function_prediction), new_propagator_state, new_decimator_state

    def _local_search(self, prediction, sat_problem, batch_replication):
        """walk-sat"""

        assignment = (prediction[0] > 0.5).float()
        assignment = sat_problem._active_variables * (2 * assignment - 1.0)

        sat_problem._edge_mask = torch.mm(sat_problem._graph_mask_tuple[1], sat_problem._active_variables) * \
                                 torch.mm(sat_problem._graph_mask_tuple[3], sat_problem._active_functions)

        for _ in range(self._local_search_iterations):
            unsat_examples, unsat_functions = self._compute_energy(assignment, sat_problem)
            unsat_examples = (unsat_examples > 0).float()

            if batch_replication > 1:
                compact_unsat_examples = 1 - (torch.mm(sat_problem._replication_mask_tuple[1], 1 - unsat_examples) >
                                              0).float()
                if compact_unsat_examples.sum() == 0:
                    break
            elif unsat_examples.sum() == 0:
                break

            delta_energy = self._compute_energy_diff(assignment, sat_problem)
            max_delta_ind = util.sparse_argmax(-delta_energy.squeeze(1), sat_problem._batch_mask_tuple[0],
                                               device = self._device)

            unsat_variables = torch.mm(sat_problem._vf_mask_tuple[0], unsat_functions) * sat_problem._active_variables
            unsat_variables = (unsat_variables > 0).float() * torch.rand([sat_problem._variable_num, 1],
                                                                         device = self._device)
            random_ind = util.sparse_argmax(unsat_variables.squeeze(1), sat_problem._batch_mask_tuple[0],
                                            device = self._device)

            coin = (torch.rand(sat_problem._batch_size, device = self._device) > self._epsilon).long()
            max_ind = coin * max_delta_ind + (1 - coin) * random_ind
            max_ind = max_ind[unsat_examples[:, 0] > 0]

            # Flipping the selected variables
            assignment[max_ind, 0] = -assignment[max_ind, 0]

        return (assignment + 1) / 2.0, prediction[1]

    def _check_recurrence_termination(self, active, prediction, sat_problem):
        """停用模型已找到SAT解决方案的CNF"""

        output, _ = self._cnf_evaluator(variable_prediction = prediction[0], graph_map = sat_problem._graph_map,
                                        batch_variable_map = sat_problem._batch_variable_map,
                                        batch_function_map = sat_problem._batch_function_map,
                                        edge_feature = sat_problem._edge_feature,
                                        meta_data = sat_problem._meta_data)  # .detach().cpu().numpy()

        if sat_problem._batch_replication > 1:
            real_batch = torch.mm(sat_problem._replication_mask_tuple[1], (output > 0.5).float())
            dup_batch = torch.mm(sat_problem._replication_mask_tuple[0], (real_batch == 0).float())
            active[active[:, 0], 0] = (dup_batch[active[:, 0], 0] > 0)
        else:
            active[active[:, 0], 0] = (output[active[:, 0], 0] <= 0.5)

    def get_init_state(self, graph_map, randomized, batch_replication = 1):
        """初始化变量子节点和子句子节点"""

        if self._propagator is None:
            init_propagator_state = None
        else:
            init_propagator_state = self._propagator.get_init_state(graph_map, randomized, batch_replication)

        if self._predictor is None:
            init_decimator_state = None
        else:
            init_decimator_state = self._predictor.get_init_state(graph_map, randomized, batch_replication)

        return init_propagator_state, init_decimator_state


class SPNueralBase:
    def __init__(self, dimacs_file, device, epoch_replication = 3, batch_replication = 1, epoch = 100, batch_size = 2000,
                 hidden_dimension = 160, feature_dim = 100, train_outer_recurrence_num = 5, alpha = 0.2, use_cuda =
                 True):
        self._use_cuda = use_cuda and torch.cuda.is_available()
        '''读入数据路径'''
        self._dimacs_file = dimacs_file

        self._epoch_replication = epoch_replication
        self._batch_replication = batch_replication
        self._epoch = epoch
        self._batch_size = batch_size
        self._hidden_dimension = 2 * hidden_dimension
        self._gru_hidden_dimension = hidden_dimension
        self._feature_dim = feature_dim
        '''以下 limit 设置是用于控制 data_loader 一次读入多少行数据'''
        self._train_batch_limit = 400000
        self._test_batch_limit = 40000
        self._max_cache_size = 100000
        self._train_outer_recurrence_num = train_outer_recurrence_num
        self._device = device
        self._num_cores = multiprocessing.cpu_count()
        torch.set_num_threads(self._num_cores)
        '''计算loss类'''
        self._loss_evaluator = solver.SatLossEvaluator(alpha = alpha, device = self._device)
        '''初始化模型列表'''
        model_list = [PropagatorDecimatorSolverBase(self._device, "SP-Nueral", feature_dim, self._gru_hidden_dimension,
                                                    pure_sp = False)]
        '''将模型放到 GPU 上运行 如果 cuda 设备可用'''
        self._model_list = [self._set_device(model) for model in model_list]

    def train(self, last_export_path_base = None, best_export_path_base = None, metric_index = 0, load_model = None,
              train_epoch_size = 0, is_train = True):

        train_loader = util.FactorGraphDataset.get_loader(
            input_file = self._dimacs_file, limit = self._train_batch_limit,
            hidden_dim = self._hidden_dimension, batch_size = self._batch_size, shuffle = True,
            num_workers = self._num_cores, max_cache_size = self._max_cache_size,
            epoch_size = train_epoch_size)

        '''设置模型保存路径 若不存在 则创建该文件夹'''
        if not os.path.exists(best_export_path_base):
            os.makedirs(best_export_path_base)

        if not os.path.exists(last_export_path_base):
            os.makedirs(last_export_path_base)
        losses = np.zeros((len(self._model_list), self._epoch, self._epoch_replication), dtype = np.float32)
        for rep in range(self._epoch_replication):
            '''# 在之前模型的基础上进行训练'''
            if load_model == "best" and best_export_path_base is not None:
                self._load(best_export_path_base)
            elif load_model == "last" and last_export_path_base is not None:
                self._load(last_export_path_base)
            for epoch in range(self._epoch):
                start_time = time.time()
                losses[:, epoch, rep] = SPModel.run_cora(train_loader, rep, epoch, self._model_list,
                                                         self._loss_evaluator, self._device, self._batch_replication,
                                                         self._hidden_dimension, self._feature_dim,
                                                         self._train_outer_recurrence_num, self._use_cuda, is_train, False)

                if self._use_cuda:
                    torch.cuda.empty_cache()
                duration = time.time() - start_time
                if last_export_path_base is not None:
                    for (i, model) in enumerate(self._model_list):
                        _module(model).save(last_export_path_base)
                message = ''
                for (i, model) in enumerate(self._model_list):
                    name = _module(model)._name
                    message += 'Step {:d}: {:s} loss={:5.5f} |'.format(_module(model)._global_step.int()[0], name,
                                                                       losses[i, epoch, rep])

                print('Rep {:2d}, Epoch {:2d}: {:s}'.format(rep + 1, epoch + 1, message))
                print('Time spent: %s seconds' % duration)
                # if best_export_path_base is not None:
                #     for (i, model) in enumerate(self._model_list):
                #         if errors[metric_index, i, epoch, rep] < best_errors[i]:
                #             best_errors[i] = errors[metric_index, i, epoch, rep]
                #             _module(model).save(best_export_path_base)

        if self._use_cuda:
            torch.backends.cudnn.benchmark = False

    def predict(self, out_file, import_path_base = None, epoch_replication = 1, is_train = False):
        test_loader = util.FactorGraphDataset.get_loader(
            input_file = self._dimacs_file, limit = self._test_batch_limit,
            hidden_dim = self._hidden_dimension, batch_size = self._batch_size, shuffle = False,
            num_workers = self._num_cores, max_cache_size = self._max_cache_size,
            batch_replication = epoch_replication)

        if import_path_base is not None:
            self._load(import_path_base)

        SPModel.run_cora(test_loader, self._model_list, self._device, epoch_replication, self._epoch,
                         self._hidden_dimension, self._feature_dim, self._batch_size, self._use_cuda,
                         is_train = is_train)

        if self._use_cuda:
            torch.cuda.empty_cache()

    def _set_device(self, model):
        """设置使用设备"""
        if self._use_cuda:
            return nn.DataParallel(model).cuda(self._device)
        return model.cpu()

    def _load(self, import_path_base):
        """加载模型"""
        for model in self._model_list:
            _module(model).load(import_path_base)

    def _save(self, export_path_base):
        """保存模型"""
        for model in self._model_list:
            _module(model).save(export_path_base)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to datasets')
    parser.add_argument('last_model_path', help='Path to save the previous model')
    parser.add_argument('best_model_path', help='Path to save the best model')
    parser.add_argument('-g', '--gpu_mode', help='Run on GPU', action='store_true')
    parser.add_argument('-p', '--predict', help='Run prediction', action='store_true')
    parser.add_argument('-l', '--load_model', help='Load the previous model')
    parser.add_argument('-lp', '--load_model_path', help='Path to load the previous model')

    args = parser.parse_args()
    try:
        if args.gpu_mode:
            device = torch.device("cuda")
            trainer = SPNueralBase(args.dataset_path, device = device)
        else:
            device = torch.device("cpu")
            trainer = SPNueralBase(args.dataset_path, device = device, use_cuda = False)

        if args.predict and args.load_model_path:
            trainer.predict(out_file = '', import_path_base = args.load_model_path)
        else:
            trainer.train(args.last_model_path, args.best_model_path)
    except:
        print(traceback.format_exc())

