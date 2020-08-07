import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np

import util


class MeanAggregator(nn.Module):

    def __init__(self, features, device, gru=False):
        super(MeanAggregator, self).__init__()
        self.features = features
        self._device = device
        self.gru = gru

    def forward(self, nodes, to_neighs, num_sample=10):
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample, )) if len(to_neigh) >= num_sample else _set(to_neigh) for
                           to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gru:
            samp_neighs = [samp_neigh + set([nodes[i]])
                           for i, samp_neigh in enumerate(samp_neighs)]
        '''去重'''
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(
            unique_nodes), device=self._device))
        '''获取变量节点的列坐标'''
        column_indices = [unique_nodes[n]
                          for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs))
                       for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        '''self.features 是 一个nn.Embedding, 其权重 weight 为 [function_num, variable_num]'''
        embed_matrix = self.features(torch.tensor(unique_nodes_list, device=self._device, dtype=torch.long))
        to_feats = mask.mm(embed_matrix)
        return to_feats


class SurveyAggregator(nn.Module):
    def __init__(self, device, hidden_dimension, embed_dimension, edge_dimension, meta_data_dimension,include_adaptors=False, pi=0.0):
        super(SurveyAggregator, self).__init__()
        self._device = device
        self._function_message_dim = 3
        self._variable_message_dim = 2
        self._eps = torch.tensor([1e-40], device=self._device)
        self._max_logit = torch.tensor([30.0], device=self._device)
        self._include_adaptors = include_adaptors
        self._pi = torch.tensor([pi], dtype=torch.float32, device=self._device)
        self._previous_function_state = None
        self._embed_dim = embed_dimension
        '''用于loss计算 temperature -> 退火 刚开始影响因子很大 后逐渐变小'''
        self._global_step = nn.Parameter(torch.tensor([0], dtype=torch.float, device=self._device), requires_grad=False)
        '''用于初始化variable_state和function_state的采样函数'''
        self._module_list = nn.ModuleList()
        self.norm_var_layer = nn.Sequential(nn.BatchNorm1d(self._embed_dim), nn.Linear(self._embed_dim, self._function_message_dim))
        self.norm_func_layer = nn.Sequential(
            nn.BatchNorm1d(self._embed_dim),
            nn.Linear(self._embed_dim, self._variable_message_dim)
        )
        '''若不使用纯sp算法计算, 则会使用GRU'''
        self._variable_rnn_cell = nn.GRUCell(
            self._function_message_dim + edge_dimension + meta_data_dimension, hidden_dimension, bias=True)
        self._function_rnn_cell = nn.GRUCell(
            self._variable_message_dim + edge_dimension + meta_data_dimension, hidden_dimension, bias=True)
        if self._include_adaptors:
            self._variable_input_projector = nn.Linear(hidden_dimension, self._variable_message_dim, bias=False)
            self._function_input_projector = nn.Linear(hidden_dimension, 1, bias=False)
            self._module_list = nn.ModuleList([self._variable_input_projector, self._function_input_projector])
        self._module_list.append(self.norm_var_layer)
        self._module_list.append(self.norm_func_layer)
        self._module_list.append(self._variable_rnn_cell)
        self._module_list.append(self._function_rnn_cell)

    def safe_log(self, x):
        """ 求log值 """
        return torch.max(x, self._eps).log()

    def safe_exp(self, x):
        """ 求指数 e^x """
        return torch.min(x, self._max_logit).exp()

    def forward(self, init_state, decimator_state, sat_problem, is_training, pure_sp=True):

        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, _, _, _ = sat_problem._batch_mask_tuple
        p_variable_mask, _, _, _ = sat_problem._pos_mask_tuple  # var_num * edge_num
        n_variable_mask, _, _, _ = sat_problem._neg_mask_tuple

        edge_num = init_state[0].size(0)
        mask = torch.ones(edge_num, 1, device=self._device)

        if len(decimator_state) == 3:
            decimator_variable_state, decimator_function_state, edge_mask = decimator_state
        else:
            decimator_variable_state, decimator_function_state = decimator_state
            edge_mask = None
        variable_state, function_state = init_state

        if self._include_adaptors:
            decimator_variable_state = F.logsigmoid(self._function_input_projector(
                decimator_variable_state))   # 子句对变量的影响通过self._function_input_projector 体现, logsigmoid < 0
        else:
            '''对子句编码 safe_log 防止变量里有无穷大值'''
            decimator_variable_state = self.safe_log(
                decimator_variable_state[:, 0]).unsqueeze(1)

        if edge_mask is not None:
            decimator_variable_state = decimator_variable_state * edge_mask

        aggregated_variable_state = torch.mm(
            function_mask, decimator_variable_state)  # (function_num, 1)
        aggregated_variable_state = torch.mm(
            function_mask_transpose, aggregated_variable_state)  # (edge_num, 1)
        # decimator_variable_state 是每个边上的信息值, 将这个值传递给子句(第一个aggregated_variable_state), 然后在传递给边(第二个aggregated_variable_state) 两者相减获得信息熵(第三个aggregated_variable_state)
        aggregated_variable_state = aggregated_variable_state - decimator_variable_state

        # mask 为1, function_state 为求 aggregated_variable_state 的指数
        function_state = mask * \
            self.safe_exp(aggregated_variable_state) + \
            (1 - mask) * function_state[:, 0].unsqueeze(1)
        '''变量向子句传递信息, 第一维表示信息值, 第二维度表示是正影响还是负影响'''
        if self._include_adaptors:
            decimator_function_state = self._variable_input_projector(
                decimator_function_state)
            decimator_function_state[:, 0] = torch.sigmoid(
                decimator_function_state[:, 0])
            decimator_function_state[:, 1] = torch.sign(
                decimator_function_state[:, 1])
        '''是否有外界影响因素, 如子句之间的关联, 没有则值为0'''
        external_force = decimator_function_state[:, 1].unsqueeze(1)
        '''计算 log 值防止 decimator_function_state 中有无穷小值'''
        decimator_function_state = self.safe_log(
            1 - decimator_function_state[:, 0]).unsqueeze(1)
        if edge_mask is not None:
            decimator_function_state = decimator_function_state * edge_mask

        '''求出正影响的变量'''
        pos = torch.mm(
            p_variable_mask, decimator_function_state)  # decimator_function_state[edge,1]
        pos = torch.mm(variable_mask_transpose, pos)  #正影响的边
        '''# 求出负影响的变量'''
        neg = torch.mm(n_variable_mask, decimator_function_state)
        neg = torch.mm(variable_mask_transpose, neg)    #负影响的边

        '''# 与 edge_feature 符号一致的变量对子句的影响'''
        same_sign = 0.5 * (1 + sat_problem._edge_feature) * \
            pos + 0.5 * (1 - sat_problem._edge_feature) * neg
        same_sign = same_sign - decimator_function_state
        same_sign += self.safe_log(1.0 - self._pi *
                                   (external_force == sat_problem._edge_feature).float())  #external_force表示是正影响还是负影响
        '''# 与 edge_feature 符号相反的变量对子句的影响'''
        opposite_sign = 0.5 * (1 - sat_problem._edge_feature) * \
            pos + 0.5 * (1 + sat_problem._edge_feature) * neg
        opposite_sign += self.safe_log(1.0 - self._pi *
                                       (external_force == -sat_problem._edge_feature).float())

        '''# 相加为对子句整体的影响'''
        dont_care = same_sign + opposite_sign

        bias = 0  # (2 * dont_care) / 3.0
        same_sign = same_sign - bias
        opposite_sign = opposite_sign - bias

        dont_care = self.safe_exp(dont_care - bias)
        same_sign = self.safe_exp(same_sign)
        opposite_sign = self.safe_exp(opposite_sign)
        '''# q_u 对应 sp 公式中的 Πuj→a=(1−∏b∈Vua(j)(1−ηb→j))∏b∈Vsa(j)(1−ηb→j)'''
        q_u = same_sign * (1 - opposite_sign)
        q_s = opposite_sign * (1 - same_sign)
        total = q_u + q_s + dont_care
        '''# 对应 sp 公式中的 ηa→i'''
        temp = torch.cat((q_u, q_s, dont_care), 1) / total
        variable_state = mask * temp + (1 - mask) * variable_state

        del mask
        function_state = torch.cat((function_state, external_force), 1)
        '''previous_function_state 对应上一次的训练结果'''
        if self._previous_function_state is not None and not is_training:
            '''比较两次的变化'''
            function_diff = (self._previous_function_state -
                             function_state[:, 0]).abs().unsqueeze(1)
            state = torch.cat((variable_state, function_state), 1)
            '''求最大值'''
            sum_diff = util.sparse_smooth_max(
                function_diff, sat_problem._graph_mask_tuple[0], self._device)
            '''求子句对变量的影响'''
            sum_diff = sum_diff * sat_problem._active_variables
            '''如果子句对变量的影响很小 经过下列运算 sum_diff 会接近无穷小'''
            sum_diff = util.sparse_max(sum_diff.squeeze(
                1), sat_problem._batch_mask_tuple[0], self._device).unsqueeze(1)
            sum_diff = torch.mm(sat_problem._batch_mask_tuple[0], sum_diff)
            '''sum_diff.sum() > 0 说明子句对变量的影响很大'''
            if sum_diff.sum() > 0:
                '''预测变量值'''
                score, _ = self.scorer(state, sat_problem)
                '''scores 为变量预测值 coeff 计算出的为 sum_diff "联合"影响'''
                coeff = score.abs() * sat_problem._active_variables * sum_diff
                '''coeff_sum 为了再次确认对变量的影响 进入下列循环表示此时可以确认某些变量的值为定值'''
                if coeff.sum() > 0:
                    max_ind = util.sparse_argmax(coeff.squeeze(
                        1), sat_problem._batch_mask_tuple[0], self._device)
                    norm = torch.mm(sat_problem._batch_mask_tuple[1], coeff)
                    max_ind = max_ind[norm.squeeze(1) != 0]
                    if max_ind.size()[0] > 0:
                        assignment = torch.zeros(
                            sat_problem._variable_num, 1, device=self._device)
                        assignment[max_ind, 0] = score.sign()[max_ind, 0]
                        sat_problem.set_variables(assignment)
            '''更新 _previous_function_state'''
            self._previous_function_state = function_state[:, 0]

        else:
            self._previous_function_state = function_state[:, 0]
        '''如果不使用纯SP算法, 则用GRU计算特征的传播'''
        if not pure_sp:
            decimator_state = self.decimate(
                decimator_state, (variable_state, function_state), sat_problem)
        return (variable_state, function_state), decimator_state

    def get_init_state(self, graph_map, randomized, batch_replication):
        self._previous_function_state = None
        edge_num = graph_map.size(1) * batch_replication

        if randomized:
            '''用norm_var_layer采样产生的data作为初始化数据'''
            # variable_state = self.norm_var_layer(torch.rand(edge_num, self._embed_dim, dtype = torch.float32))
            # function_state = self.norm_func_layer(torch.rand(edge_num, self._embed_dim, dtype = torch.float32))
            variable_state = torch.rand(edge_num, self._function_message_dim, dtype=torch.float32,
                                        device=self._device)
            function_state = torch.rand(edge_num, self._variable_message_dim, dtype=torch.float32,
                                        device=self._device)
            '''第二列数据用于表示外来影响因素(目前模型中没有,因此值为空)'''
            function_state[:, 1] = 0
        else:
            '''将所有值初始化为1'''
            variable_state = torch.ones(edge_num, self._function_message_dim, dtype=torch.float32,
                                        device=self._device) / self._function_message_dim
            function_state = 0.5 * torch.ones(edge_num, self._variable_message_dim, dtype=torch.float32,
                                              device=self._device)
            function_state[:, 1] = 0

        return (variable_state, function_state)

    def scorer(self, function_message, sat_problem, last_call=False):
        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, _, _, _ = sat_problem._batch_mask_tuple
        p_variable_mask, _, _, _ = sat_problem._pos_mask_tuple
        n_variable_mask, _, _, _ = sat_problem._neg_mask_tuple
        """是否有外界影响因素, 如子句之间的关联, 没有则值为0"""
        external_force = torch.sign(
            torch.mm(variable_mask, function_message[:, 1].unsqueeze(1)))
        function_message = self.safe_log(
            1 - function_message[:, 0]).unsqueeze(1)
        """所有未确定值的边 -> 确定值表示只能取该值 否则矛盾"""
        edge_mask = torch.mm(function_mask_transpose,
                             sat_problem._active_functions)
        function_message = function_message * edge_mask
        """pos neg dont_care 含义同上 forward 方法中的值"""
        pos = torch.mm(p_variable_mask, function_message) + self.safe_log(
            1.0 - self._pi * (external_force == 1).float())
        neg = torch.mm(n_variable_mask, function_message) + self.safe_log(
            1.0 - self._pi * (external_force == -1).float())

        pos_neg_sum = pos + neg

        dont_care = torch.mm(variable_mask, function_message) + \
            self.safe_log(1.0 - self._pi)

        bias = (2 * pos_neg_sum + dont_care) / 4.0
        pos = pos - bias
        neg = neg - bias
        pos_neg_sum = pos_neg_sum - bias
        dont_care = self.safe_exp(dont_care - bias)

        '''q_0 对应 sp 计算公式中的 Π0j→a'''
        q_0 = self.safe_exp(pos) - self.safe_exp(pos_neg_sum)
        q_1 = self.safe_exp(neg) - self.safe_exp(pos_neg_sum)

        total = self.safe_log(q_0 + q_1 + dont_care)

        return self.safe_exp(self.safe_log(q_1) - total) - self.safe_exp(self.safe_log(q_0) - total), None

    def decimate(self, init_state, message_state, sat_problem, active_mask=None):
        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose = \
            sat_problem._batch_mask_tuple

        if active_mask is not None:
            mask = torch.mm(b_variable_mask, active_mask.float())
            mask = torch.mm(variable_mask_transpose, mask)
        else:
            edge_num = init_state[0].size(0)
            mask = torch.ones(edge_num, 1, device=self._device)

        if len(sat_problem._meta_data[0]) > 0:
            graph_feat = torch.sparse.FloatTensor(b_variable_mask._indices(), torch.tensor(np.concatenate(
                np.array(sat_problem._meta_data)), dtype=torch.float)).to_dense()
            graph_feat = torch.sum(
                torch.mm(variable_mask_transpose, graph_feat), 1).unsqueeze(1)

        variable_state, function_state = message_state

        '''变量向量表示'''
        variable_state = torch.cat(
            (variable_state, sat_problem._edge_feature), 1)

        if len(sat_problem._meta_data[0]) > 0:
            variable_state = torch.cat((variable_state, graph_feat), 1)
        else:
            variable_mask = torch.mm(
                sat_problem._signed_mask_tuple[1].to_dense(), sat_problem._solution.unsqueeze(1))
            variable_state = torch.cat((variable_state, variable_mask), 1)

        variable_state = mask * \
            self._variable_rnn_cell(
                variable_state, init_state[0]) + (1 - mask) * init_state[0]

        '''子句向量表示'''
        function_state = torch.cat(
            (function_state, sat_problem._edge_feature), 1)

        if len(sat_problem._meta_data[0]) > 0:
            function_state = torch.cat((function_state, graph_feat), 1)
        else:
            function_mask = torch.mm(
                sat_problem._signed_mask_tuple[3].to_dense(), sat_problem._active_functions)
            function_state = torch.cat((function_state, function_mask), 1)

        function_state = mask * \
            self._function_rnn_cell(
                function_state, init_state[1]) + (1 - mask) * init_state[1]

        del mask

        return variable_state, function_state


class PermutationAggregator(nn.Module):
    """多层神经网络模拟实现置换不变的效果"""

    def __init__(self, device, input_dimension, output_dimension, mem_hidden_dimension,
                 mem_agg_hidden_dimension, agg_hidden_dimension, feature_dimension):

        super(PermutationAggregator, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()

        self._W1_m = nn.Linear(
            input_dimension, mem_hidden_dimension, bias=True)  # .to(self._device)

        self._W2_m = nn.Linear(
            mem_hidden_dimension, mem_agg_hidden_dimension, bias=False)  # .to(self._device)

        self._module_list.append(self._W1_m)
        self._module_list.append(self._W2_m)

        if mem_hidden_dimension <= 0:
            mem_agg_hidden_dimension = input_dimension

        self._W1_a = nn.Linear(
            mem_agg_hidden_dimension, agg_hidden_dimension, bias=True)  # .to(self._device)

        self._W2_a = nn.Linear(
            agg_hidden_dimension, output_dimension, bias=False)  # .to(self._device)

        self._module_list.append(self._W1_a)
        self._module_list.append(self._W2_a)

        self._agg_hidden_dimension = agg_hidden_dimension
        self._mem_hidden_dimension = mem_hidden_dimension
        self._mem_agg_hidden_dimension = mem_agg_hidden_dimension

    def forward(self, state, mask, mask_transpose, edge_mask=None):

        # Apply the pre-aggregation transform
        state = F.logsigmoid(self._W2_m(F.logsigmoid(self._W1_m(state))))
        '''edge_mask默认为空'''
        if edge_mask is not None:
            state = state * edge_mask
        '''mask为当前所有有效的变量, 默认是值全部为1的向量'''
        aggregated_state = torch.mm(mask, state)

        aggregated_state = F.logsigmoid(self._W2_a(
            F.logsigmoid(self._W1_a(aggregated_state))))

        return aggregated_state


class SurveyNeuralPredictor(nn.Module):
    """使用神经网络的方法来预测CNF的解"""

    def __init__(self, device, decimator_dimension, prediction_dimension, edge_dimension, meta_data_dimension,
                 mem_hidden_dimension, agg_hidden_dimension, mem_agg_hidden_dimension, variable_classifier=None):

        super(SurveyNeuralPredictor, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()
        '''将变量分为1, -1两类'''
        self._variable_classifier = variable_classifier
        self._hidden_dimension = decimator_dimension

        if variable_classifier is not None:
            """置换不变的多层神经网络"""
            self._variable_aggregator = PermutationAggregator(device, decimator_dimension + edge_dimension +
                                                              meta_data_dimension, decimator_dimension,
                                                              mem_hidden_dimension, mem_agg_hidden_dimension,
                                                              agg_hidden_dimension, meta_data_dimension)

            self._module_list.append(self._variable_aggregator)
            self._module_list.append(self._variable_classifier)

    def forward(self, decimator_state, sat_problem, last_call=False):

        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose =\
            sat_problem._batch_mask_tuple

        variable_prediction = None

        if len(sat_problem._meta_data[0]) > 0:
            graph_feat = torch.sparse.FloatTensor(b_variable_mask._indices(), torch.tensor(np.concatenate(
                np.array(sat_problem._meta_data)), device=self._device, dtype=torch.float)).to_dense()
            graph_feat = torch.sum(
                torch.mm(variable_mask_transpose, graph_feat), 1).unsqueeze(1)

        if len(decimator_state) == 3:
            variable_state, _, edge_mask = decimator_state
        else:
            variable_state, _ = decimator_state
            edge_mask = None

        if self._variable_classifier is not None:
            '''将变量与边向量拼接'''
            aggregated_variable_state = torch.cat(
                (variable_state, sat_problem._edge_feature), 1)

            if len(sat_problem._meta_data[0]) > 0:
                aggregated_variable_state = torch.cat(
                    (aggregated_variable_state, graph_feat), 1)
            else:
                aggregated_variable_mask = torch.mm(sat_problem._signed_mask_tuple[1].to_dense(),
                                                    sat_problem._solution.unsqueeze(1))
                aggregated_variable_state = torch.cat(
                    (aggregated_variable_state, aggregated_variable_mask), 1)
            '''应用置换不变的聚合器'''
            aggregated_variable_state = self._variable_aggregator(
                aggregated_variable_state, variable_mask, variable_mask_transpose, edge_mask)

            variable_prediction = self._variable_classifier(
                aggregated_variable_state)

        return variable_prediction

    def get_init_state(self, graph_map, randomized, batch_replication):

        edge_num = graph_map.size(1) * batch_replication

        if randomized:
            variable_state = 2.0 * torch.rand(edge_num, self._hidden_dimension, dtype=torch.float32,device=self._device) - 1.0
            function_state = 2.0 * torch.rand(edge_num, self._hidden_dimension, dtype=torch.float32,device=self._device) - 1.0
        else:
            variable_state = torch.zeros(edge_num, self._hidden_dimension, dtype=torch.float32, device=self._device)
            function_state = torch.zeros(edge_num, self._hidden_dimension, dtype=torch.float32, device=self._device)
        return (variable_state, function_state)