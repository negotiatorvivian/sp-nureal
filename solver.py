import torch
import torch.nn as nn
import numpy as np


class IdentityPredictor(nn.Module):
    def __init__(self, device, random_fill = False):
        """用于预测 SAT 变量取值"""
        super(IdentityPredictor, self).__init__()
        self._random_fill = random_fill
        self._device = device

    def _update_solution(self, prediction, sat_problem):
        if prediction[0] is not None:
            '''_active_variables 的值为prediction[0] 非_active_variables 值已经确定 对应 sat_problem._solution'''
            variable_solution = sat_problem._active_variables * prediction[0] + \
                                (1.0 - sat_problem._active_variables) * sat_problem._solution.unsqueeze(1)
            '''更新已经确定的变量对应的 _active_variables 为 0 -> 变为不活跃的变量'''
            sat_problem._solution[sat_problem._active_variables[:, 0] == 1] = \
                variable_solution[sat_problem._active_variables[:, 0] == 1, 0]
        else:
            variable_solution = None

        return variable_solution, prediction[1]

    def forward(self, decimator_state, sat_problem, last_call = False):
        pred = sat_problem._solution.unsqueeze(1)

        if self._random_fill and last_call:
            active_var_num = (sat_problem._active_variables[:, 0] > 0).long().sum()

            if active_var_num > 0:
                pred[sat_problem._active_variables[:, 0] > 0, 0] = \
                    torch.rand(active_var_num.item(), device = self._device)

        prediction = self._update_solution(pred, sat_problem)
        return prediction


class SatCNFEvaluator(nn.Module):
    """判断当前 CNF 的解是否正确"""

    def __init__(self, device):
        super(SatCNFEvaluator, self).__init__()
        self._device = device
        self._variable = None
        self._unsat = None
        self._sat = None
        self._unsat_limit = 5
        self._count = 0

    def forward(self, variable_prediction, graph_map, batch_variable_map,
                batch_function_map, edge_feature, vf_mask = None, graph_mask = None, active_variables = None,
                active_functions = None):
        # self._variable = torch.sign(variable_prediction)
        function_num = batch_function_map.size(0)
        all_ones = torch.ones(function_num, 1, device = self._device)

        signed_variable_mask_transpose, function_mask = \
            SatLossEvaluator.compute_masks(graph_map, batch_variable_map, batch_function_map, edge_feature,
                                           self._device)

        b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose = \
            SatLossEvaluator.compute_batch_mask(batch_variable_map, batch_function_map, self._device)

        # edge_values = torch.mm(signed_variable_mask_transpose, self._variable)
        edge_values = torch.mm(signed_variable_mask_transpose, variable_prediction)
        edge_values = edge_values + (1 - edge_feature) / 2
        edge_values = (edge_values > 0.5).float()

        clause_values = torch.mm(function_mask, edge_values)
        clause_values = (clause_values > 0).float()
        if vf_mask is not None:
            '''所有出现在不满足的子句中的变量并去重'''
            unsat_vars = vf_mask.to_dense()[np.argwhere(clause_values.squeeze(1) == 0)[0]]
            unsat_vars = set(np.argwhere(unsat_vars > 0)[1].numpy())
            '''计算上一次结果与这一次结果中不满足的变量的差异, 若小于_unsat_limit 且连续3次均如此 -> 本轮训练不需要再继续进行了'''
            if self._unsat is not None and len(self._unsat - unsat_vars) < self._unsat_limit:
                self._count += 1
                # if self._count >= 3:
                #     return None, None
            self._unsat = unsat_vars
            '''所有出现在满足中的子句中的变量并去重'''
            sat_vars = vf_mask.to_dense()[np.argwhere(clause_values.squeeze(1) == 1)[0]]
            sat_vars = set(np.argwhere(sat_vars > 0)[1].numpy())
            '''相减后获得差集: 即可以确定值的变量 -> 变量所在的子句均为可满足子句'''
            self._sat = sat_vars - self._unsat
            functions = set(np.argwhere(vf_mask.to_dense()[:, list(self._sat)])[0].numpy()) - set(
                np.argwhere(vf_mask.to_dense()[:, list(unsat_vars)] > 0)[0].numpy())
            functions = list(functions) if functions is not None else []
            variables = list(set(np.argwhere(vf_mask.to_dense()[functions, :] > 0)[1].numpy()))

        '''最大可满足子句数'''
        max_sat = torch.mm(b_function_mask_transpose, all_ones)
        '''实际满足的子句数'''
        batch_values = torch.mm(b_function_mask_transpose, clause_values)

        if vf_mask is not None:
            return ((max_sat == batch_values).float(), max_sat - batch_values, graph_map, clause_values), \
                   (list(self._unsat), functions, variables)
        else:
            return ((max_sat == batch_values).float(), max_sat - batch_values, graph_map, clause_values), None


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
        max_val = torch.max(x[np.argwhere(torch.isfinite(x.squeeze()) == 1)[0]])
        '''torch.clamp 用于将 x 控制在 [0, max_val.data] 区间内'''
        x = torch.clamp(x, 0, max_val.data).clone().detach().requires_grad_(True)
        a = torch.max(x, eps)
        loss = a.log()
        return loss

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
