import math
import random

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

import util

import warnings
warnings.filterwarnings("ignore")


class SATProblem(object):

    def __init__(self, data_batch, device, batch_replication = 1):
        self._device = device
        self.batch_replication = batch_replication
        self.setup_problem(data_batch, batch_replication)
        self._edge_mask = None
        self._is_sat = 0.5 * torch.ones(self._batch_size, device = self._device)
        self._cnf_evaluator = SatCNFEvaluator(device)
        self.statistics = np.zeros(3)

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

        ind = torch.arange(batch_replication, dtype = torch.int32, device = self._device).unsqueeze(1). \
            repeat(1, variable_num).view(1, -1)
        batch_variable_map = batch_variable_map.repeat(batch_replication) + ind * batch_size
        ind = torch.arange(batch_replication, dtype = torch.int32, device = self._device).unsqueeze(1). \
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
        # adj_lists = defaultdict(set)
        adj_lists = {}
        node_list = []
        self.nodes = ((self._signed_mask_tuple[0]._indices()[0].to(torch.float) + 1) * self._edge_feature.squeeze(1)) \
            .to(torch.long)
        for j in range(self._variable_num):
            indices = self._graph_map[1][torch.abs(self.nodes) == j + 1].to(torch.long)
            # functions = np.array(self._vf_mask_tuple[3].to(torch.long).to_dense()[indices, :][:, j] * (indices + 1))
            functions = self._vf_mask_tuple[3].to(torch.long).to_dense()[indices.cpu().numpy(), :][:, j] * (indices + 1)
            node_list.append(functions)
            edge_indices = np.argwhere(self._signed_mask_tuple[2].cpu().to_dense()[indices] != 0)[1]
            # relations = set([abs(i) - 1 for i in self.nodes[edge_indices].numpy()])
            relations = torch.unique(torch.abs(self.nodes[edge_indices]) - 1, sorted = False)

            if len(relations) < 2:
                continue
            # adj_lists[j] = relations - set([j])
            adj_lists[j] = relations
        return adj_lists, node_list

    def _post_process_predictions(self, prediction, is_training = True):
        """计算 cnf 中可满足的子句数"""
        output, res, activations = self._cnf_evaluator(prediction, self._graph_map, self._batch_variable_map,
                                                       self._batch_function_map, self._edge_feature,
                                                       self._vf_mask_tuple[1], self._graph_mask_tuple[3],
                                                       self._active_variables, self._active_functions, self,
                                                       is_training)

        if res is None:
            return None
        # unsat_clause_num, graph_map, clause_values = [a.detach().cpu().numpy() for a in res]
        if activations is not None:
            trained_vars, certain_vars = activations
            print('sat:', output, '\tcertain_vars:', certain_vars)
            '''若res不为空, trained_vars表示所有不满足子句中出现过的变量'''
            return trained_vars, output, certain_vars


class SatCNFEvaluator(nn.Module):
    """判断当前 CNF 的解是否正确"""

    def __init__(self, device):
        super(SatCNFEvaluator, self).__init__()
        self._device = device
        self._unsat = None
        self._sat = None
        self._increment = 0.6
        self._floor = nn.Parameter(torch.tensor([1], dtype = torch.float, device = self._device), requires_grad = False)
        self._temperature = nn.Parameter(torch.tensor([2], dtype = torch.float, device = self._device),
                                         requires_grad = False)

    def forward(self, variable_prediction, graph_map, batch_variable_map, batch_function_map, edge_feature,
                vf_mask = None, graph_mask = None, active_variables = None, active_functions = None, sat_problem = None,
                is_training = True):
        function_num = batch_function_map.size(0)
        all_ones = torch.ones(function_num, 1, device = self._device)

        signed_variable_mask_transpose, function_mask = \
            SatLossEvaluator.compute_masks(graph_map, batch_variable_map, batch_function_map, edge_feature,
                                           self._device)

        b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose = \
            SatLossEvaluator.compute_batch_mask(batch_variable_map, batch_function_map, self._device)

        edge_values = torch.mm(signed_variable_mask_transpose, variable_prediction)
        edge_values = edge_values + (1 - edge_feature) / 2
        edge_values = (edge_values > 0.5).float()

        clause_values = torch.mm(function_mask, edge_values)
        clause_values = (clause_values > 0).float()
        res = False
        if vf_mask is not None:
            vf_mask_ = vf_mask.cpu()
            clause_values_ = clause_values.cpu()
            '''所有出现在不满足的子句中的变量并去重'''
            unsat_vars = vf_mask_.to_dense()[np.argwhere(clause_values_.squeeze(1) == 0)[0]]
            unsat_vars = set(np.argwhere(unsat_vars > 0)[1].numpy())
            self._unsat = unsat_vars
            '''所有出现在满足中的子句中的变量并去重'''
            sat_vars = vf_mask_.to_dense()[np.argwhere(clause_values_.squeeze(1) == 1)[0]]
            sat_vars = set(np.argwhere(sat_vars > 0)[1].numpy())
            '''相减后获得差集: 即可以确定值的变量 -> 变量所在的子句均为可满足子句'''
            self._sat = sat_vars - self._unsat
            res, variables = self.simplify(sat_problem, variable_prediction, is_training)

        '''最大可满足子句数'''
        max_sat = torch.mm(b_function_mask_transpose, all_ones)
        '''实际满足的子句数'''
        batch_values = torch.mm(b_function_mask_transpose, clause_values)

        if vf_mask is not None:
            return res, ((max_sat == batch_values).float(), graph_map, clause_values), (list(self._unsat), variables)
        else:
            return res, ((max_sat == batch_values).float(), max_sat - batch_values, graph_map, clause_values), \
                   None

    def simplify(self, sat_problem, variable_prediction, is_training):
        variables = list(self._sat)
        # functions = np.array([l.numpy() for l in sat_problem.node_adj_lists])[variables]
        functions = np.array(sat_problem.node_adj_lists)[variables]
        symbols = ((variable_prediction[variables] > 0.5).to(torch.float) * 2 - 1).to(torch.long)
        try_times = 5
        ending = ''
        function_num_addition = 0
        flag = 1
        # indices = random.sample(range(len(variables)), math.floor(math.pow(self._temperature, self._increment)))
        if not is_training:
            try_times = 10
        print('\n----------------------')
        while try_times > 0:
            sample_num = max(math.floor(math.pow(self._temperature, self._increment)), 1)
            sample_num = min(sample_num, len(variables))
            indices = random.sample(range(len(variables)), sample_num)
            deactivate_functions = []
            deactivate_varaibles = []
            for j in range(len(indices)):
                i = indices[j]
                symbols_ = symbols * flag
                pos_functions = np.array(functions[i][torch.tensor(functions[i]) * symbols_[i] > 0].cpu()).flatten()
                # pos_functions = np.array(functions[i][torch.tensor(functions[i]) * symbols[i] > 0].
                #                          to(self._device)).flatten()

                if len(pos_functions) < len(functions[i]):
                    deactivate_varaibles.append(variables[i])
                deactivate_functions.extend(np.abs(pos_functions) - 1)
            deactivate_functions = list(set(deactivate_functions)) if len(deactivate_functions) > 0 else []
            sat_str = 'p cnf ' + str(sat_problem._variable_num) + ' ' + \
                      str(sat_problem._function_num - len(deactivate_functions) + function_num_addition) + '\n'
            for j in range(sat_problem._function_num):
                if j not in deactivate_functions:
                    clause = ((sat_problem._graph_map[0] + 1) * sat_problem._edge_feature.squeeze().to(torch.int))[
                        sat_problem._graph_map[1] == j]
                    function_str = [i for i in map(str, clause.cpu().numpy()) if abs(int(i)) - 1 not in deactivate_varaibles]
                    if len(function_str) == 0:
                        return False, None
                    sat_str += ' '.join(function_str)
                    sat_str += ' 0\n'
            sat_str += ending

            print('temperature: ', self._temperature)
            res = util.use_solver(sat_str)
            if res:
                self._temperature += 1
                sat_problem.statistics[0] += 1
                return res, (np.array(variables)[indices] + 1, (symbols[indices].squeeze() > 0))
            elif res is False:
                sat_problem.statistics[1] += 1
                try_times -= 1
                print(sat_problem.statistics, try_times)
                # self._temperature += 0.5
                unsat_condition = (np.array(deactivate_varaibles) + 1) * np.array(symbols[indices].cpu()).flatten() * -1
                ending += ' '.join([str(i) for i in unsat_condition])
                ending += ' 0\n'
                function_num_addition += 1

            else:
                if self._temperature > 0:
                    self._temperature -= 1
                try_times -= 1
                sat_problem.statistics[2] += 1
                if sat_problem.statistics[2] >= 3:
                    flag = -1
                if try_times > 0:
                    for item in deactivate_varaibles:
                        variables.remove(item)

        return res, (np.array(variables)[indices] + 1, (symbols[indices].squeeze() > 0))


class SatLossEvaluator(nn.Module):
    def __init__(self, alpha, device):
        """计算 loss loss function -> 能量函数: 负对数概率函数 -logP(X)"""
        super(SatLossEvaluator, self).__init__()
        self._alpha = alpha
        self._device = device
        self._max_coeff = 10.0
        self._loss_sharpness = 5

    @staticmethod
    def safe_log(x, eps):
        """计算 log 值"""
        # max_val = torch.max(x[np.argwhere(torch.isfinite(x.squeeze()) == 1)[0]])
        # '''torch.clamp 用于将 x 控制在 [0, max_val.data] 区间内'''
        # x = torch.clamp(x, 0, max_val.data).clone().detach().requires_grad_(True)
        # a = torch.max(x, eps)
        # loss = a.log()
        return torch.tensor(torch.max(x, eps).log(), requires_grad=True)

    @staticmethod
    def compute_masks(graph_map, batch_variable_map, batch_function_map, edge_feature, device):
        edge_num = graph_map.size(1)
        variable_num = batch_variable_map.size(0)
        function_num = batch_function_map.size(0)
        all_ones = torch.ones(edge_num, device = device)
        edge_num_range = torch.arange(edge_num, dtype = torch.int64, device = device)
        '''获取 CNF 的变量(每一行的叠加)'''
        variable_sparse_ind = torch.stack([edge_num_range, graph_map[0, :].long()])
        function_sparse_ind = torch.stack([graph_map[1, :].long(), edge_num_range])

        if device.type == 'cuda':
            '''获取对变量节点的编码 -> 就是将变量值转化为稀疏矩阵方便计算'''
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, edge_feature.squeeze(1),
                                                          torch.Size([edge_num, variable_num]), device = device)
            '''获取对子句节点的编码 -> 就是将子句值转化为稀疏矩阵方便计算'''
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, all_ones,
                                                          torch.Size([function_num, edge_num]), device = device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, edge_feature.squeeze(1),
                                                     torch.Size([edge_num, variable_num]), device = device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, all_ones,
                                                     torch.Size([function_num, edge_num]), device = device)

        return variable_mask, function_mask

    @staticmethod
    def compute_batch_mask(batch_variable_map, batch_function_map, device):
        """与上述方法功能相同, 区别在与batch: batch > 1 表示读数据时同一个 CNF 读入了 batch 遍"""
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]
        variable_all_ones = torch.ones(variable_num, device = device)
        function_all_ones = torch.ones(function_num, device = device)
        variable_range = torch.arange(variable_num, dtype = torch.int64, device = device)
        function_range = torch.arange(function_num, dtype = torch.int64, device = device)
        batch_size = (batch_variable_map.max() + 1).long().item()

        variable_sparse_ind = torch.stack([variable_range, batch_variable_map.long()])
        function_sparse_ind = torch.stack([function_range, batch_function_map.long()])

        if device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, variable_all_ones,
                                                          torch.Size([variable_num, batch_size]), device = device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, function_all_ones,
                                                          torch.Size([function_num, batch_size]), device = device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, variable_all_ones,
                                                     torch.Size([variable_num, batch_size]), device = device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, function_all_ones,
                                                     torch.Size([function_num, batch_size]), device = device)

        variable_mask_transpose = variable_mask.transpose(0, 1)
        function_mask_transpose = function_mask.transpose(0, 1)
        return (variable_mask, variable_mask_transpose, function_mask, function_mask_transpose)

    def forward(self, variable_prediction, label, graph_map, batch_variable_map, batch_function_map, edge_feature,
                meta_data, global_step, eps):
        """temperature"""
        coeff = torch.min(global_step.pow(self._alpha), torch.tensor([self._max_coeff], device = self._device))
        '''计算带有符号的变量与子句的编码'''
        signed_variable_mask_transpose, function_mask = \
            SatLossEvaluator.compute_masks(graph_map, batch_variable_map, batch_function_map,
                                           edge_feature, self._device)
        '''变量节点的edge_feature * variable_prediction 求出此时子句对变量实际的影响'''
        edge_values = torch.mm(signed_variable_mask_transpose, variable_prediction)
        '''如果 edge_feature = 1, edge_value 保留原始的预测值; 否则取值为(1 - prediction)'''
        edge_values = edge_values + (1 - edge_feature) / 2
        '''权重为 e^(coeff * edge_values)'''
        weights = (coeff * edge_values).exp()

        '''平滑可微的求最大值方法  -> 最大值对应求逻辑表达式的析取值最大'''
        nominator = torch.mm(function_mask, weights * edge_values)
        denominator = torch.mm(function_mask, weights)
        clause_value = denominator / torch.max(nominator, eps)
        clause_value = 1 + (clause_value - 1).pow(self._loss_sharpness)
        '''safe_log 操作防止出现无穷大值'''
        return torch.mean(SatLossEvaluator.safe_log(clause_value, eps))
