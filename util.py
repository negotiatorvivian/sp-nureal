import linecache, json
import collections
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F


class DynamicBatchDivider(object):

    def __init__(self, limit, hidden_dim):
        self.limit = limit
        self.hidden_dim = hidden_dim

    def divide(self, variable_num, function_num, graph_map, edge_feature, graph_feature, label, misc_data):
        batch_size = len(variable_num)
        edge_num = [len(n) for n in edge_feature]

        graph_map_list = []
        edge_feature_list = []
        graph_feature_list = []
        variable_num_list = []
        function_num_list = []
        label_list = []
        misc_data_list = []
        '''self.limit // (max(edge_num) * self.hidden_dim 表示一次最多可以读入的数据行数'''
        if (self.limit // (max(edge_num) * self.hidden_dim)) >= batch_size:
            if graph_feature[0] is None:
                graph_feature_list = [[None]]
            else:
                graph_feature_list = [graph_feature]

            graph_map_list = [graph_map]
            edge_feature_list = [edge_feature]
            variable_num_list = [variable_num]
            function_num_list = [function_num]
            label_list = [label]
            misc_data_list = [misc_data]

        else:

            indices = sorted(range(len(edge_num)), reverse=True, key=lambda k: edge_num[k])
            sorted_edge_num = sorted(edge_num, reverse=True)

            i = 0

            while i < batch_size:
                allowed_batch_size = self.limit // (sorted_edge_num[i] * self.hidden_dim)
                if allowed_batch_size == 0:
                    allowed_batch_size = 1   # 设置一次读的数据
                ind = indices[i:min(i + allowed_batch_size, batch_size)]

                if graph_feature[0] is None:
                    graph_feature_list += [[None]]
                else:
                    graph_feature_list += [[graph_feature[j] for j in ind]]

                edge_feature_list += [[edge_feature[j] for j in ind]]
                variable_num_list += [[variable_num[j] for j in ind]]
                function_num_list += [[function_num[j] for j in ind]]
                graph_map_list += [[graph_map[j] for j in ind]]
                label_list += [[label[j] for j in ind]]
                misc_data_list += [[misc_data[j] for j in ind]]

                i += allowed_batch_size

        return variable_num_list, function_num_list, graph_map_list, edge_feature_list, graph_feature_list, label_list, misc_data_list


###############################################################


class FactorGraphDataset(data.Dataset):
    """将 CNF 文件转换为json格式 返回的是一个迭代器 data_loader"""

    def __init__(self, input_file, limit, hidden_dim, max_cache_size=100000, epoch_size=0, batch_replication=3):

        self._cache = collections.OrderedDict()
        self._epoch_size = epoch_size
        self._input_file = input_file
        self._max_cache_size = max_cache_size

        with open(self._input_file, 'r') as fh_input:
            self._row_num = len(fh_input.readlines())

        self.batch_divider = DynamicBatchDivider(limit // batch_replication, hidden_dim)

    def __len__(self):
        return self._row_num

    def __getitem__(self, idx):

        if idx in self._cache:
            return self._cache[idx]

        line = linecache.getline(self._input_file, idx + 1)
        result = self._convert_line(line)

        if len(self._cache) >= self._max_cache_size:
            self._cache.popitem(last=False)

        self._cache[idx] = result
        return result

    def _convert_line(self, json_str):
        input_data = json.loads(json_str)
        variable_num, function_num = input_data[0]

        variable_ind = np.abs(np.array(input_data[1], dtype=np.int32)) - 1
        function_ind = np.abs(np.array(input_data[2], dtype=np.int32)) - 1
        edge_feature = np.sign(np.array(input_data[1], dtype=np.float32))
        graph_map = np.stack((variable_ind, function_ind))

        answers = []
        if len(input_data) > 4:
            misc_data = np.array(input_data[4])
            if len(misc_data) > 0:
                answers = np.array([-1] * variable_num)
                answers[misc_data] = 1
            else:
                answers = misc_data
        result = input_data[3]

        return (variable_num, function_num, graph_map, edge_feature, None, float(result), answers)

    def dag_collate_fn(self, input_data):
        """数据迭代器"""
        vn, fn, gm, ef, gf, l, md = zip(*input_data)

        variable_num, function_num, graph_map, edge_feature, graph_feat, label, misc_data = \
            self.batch_divider.divide(vn, fn, gm, ef, gf, l, md)
        segment_num = len(variable_num)

        graph_feat_batch = []
        graph_map_batch = []
        batch_variable_map_batch = []
        batch_function_map_batch = []
        edge_feature_batch = []
        label_batch = []

        for i in range(segment_num):
            '''graph features叠加'''
            graph_feat_batch += [None if graph_feat[i][0] is None else torch.from_numpy(np.stack(graph_feat[i])).float()]

            '''edge feature叠加'''
            edge_feature_batch += [torch.from_numpy(np.expand_dims(np.concatenate(edge_feature[i]), 1)).float()]

            '''标签叠加'''
            label_batch += [torch.from_numpy(np.expand_dims(np.array(label[i]), 1)).float()]

            g_map_b = np.zeros((2, 0), dtype=np.int32)
            v_map_b = np.zeros(0, dtype=np.int32)
            f_map_b = np.zeros(0, dtype=np.int32)
            variable_ind = 0
            function_ind = 0

            for j in range(len(graph_map[i])):
                graph_map[i][j][0, :] += variable_ind
                graph_map[i][j][1, :] += function_ind
                g_map_b = np.concatenate((g_map_b, graph_map[i][j]), axis=1)

                v_map_b = np.concatenate((v_map_b, np.tile(j, variable_num[i][j])))
                f_map_b = np.concatenate((f_map_b, np.tile(j, function_num[i][j])))

                variable_ind += variable_num[i][j]
                function_ind += function_num[i][j]

            graph_map_batch += [torch.from_numpy(g_map_b).int()]
            batch_variable_map_batch += [torch.from_numpy(v_map_b).int()]
            batch_function_map_batch += [torch.from_numpy(f_map_b).int()]

        return graph_map_batch, batch_variable_map_batch, batch_function_map_batch, edge_feature_batch, graph_feat_batch, label_batch, misc_data, variable_num, function_num

    @staticmethod
    def get_loader(input_file, limit, hidden_dim, batch_size, shuffle, num_workers,
                    max_cache_size=100000, use_cuda=True, epoch_size=0, batch_replication=1):

        dataset = FactorGraphDataset(
            input_file=input_file,
            limit=limit,
            hidden_dim=hidden_dim,
            max_cache_size=max_cache_size,
            epoch_size=epoch_size,
            batch_replication=batch_replication)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.dag_collate_fn,
            pin_memory=use_cuda)

        return data_loader


class CompactDimacs:
    def __init__(self, dimacs_file, to_file = False, output_file = None):
        """将 CNF 文件转换为 json 格式"""
        with open(dimacs_file, 'r') as f:
            j = 0
            for line in f:
                seg = line.split(" ")
                '''首字符为 c 跳过'''
                if seg[0] == 'c':
                    continue

                if seg[0] == 'p':
                    var_num = int(seg[2])
                    clause_num = int(seg[3])
                    self._clause_mat = np.zeros((clause_num, var_num), dtype = np.int32)

                elif len(seg) <= 1:
                    continue
                else:
                    temp = np.array(seg[:-2], dtype = np.int32)
                    self._clause_mat[j, np.abs(temp) - 1] = np.sign(temp)
                    j += 1

        ind = np.where(np.sum(np.abs(self._clause_mat), 1) > 0)[0]
        self._clause_mat = self._clause_mat[ind, :]
        if to_file and output_file:
            self.to_json(output_file)

    def to_json(self, output_file):
        clause_num, var_num = self._clause_mat.shape

        ind = np.nonzero(self._clause_mat)
        content = [[var_num, clause_num], list((ind[1] + 1) * self._clause_mat[ind]), list(ind[0] + 1)]
        with open(output_file, 'w') as f:
            f.write(str(content).replace("'", '"') + '\n')


class Perceptron(nn.Module):
    """单层神经网络"""

    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(Perceptron, self).__init__()
        self._layer1 = nn.Linear(input_dimension, hidden_dimension)
        self._layer2 = nn.Linear(hidden_dimension, output_dimension, bias=False)

    def forward(self, inp):
        return torch.sigmoid(self._layer2(F.relu(self._layer1(inp))))


def update_solution(prediction, sat_problem):
    """根据当前预测更新SAT问题对象的解"""
    if torch.sum(sat_problem._active_variables) == 0:
        print('当前激活的变量个数为 0')
    if prediction is not None:
        variable_solution = sat_problem._active_variables * prediction + \
                            (1.0 - sat_problem._active_variables) * sat_problem._solution.unsqueeze(1)
        sat_problem._solution[sat_problem._active_variables[:, 0] == 1] = \
            variable_solution[sat_problem._active_variables[:, 0] == 1, 0]
    else:
        variable_solution = None

    return variable_solution


def sparse_smooth_max(x, mask, device, alpha = 30):
    """对一个稀疏矩阵求最大值(非精确)"""
    coeff = safe_exp(alpha * x, device)
    return torch.mm(mask, x * coeff) / torch.max(torch.mm(mask, coeff), torch.ones(1, device = device))


def safe_exp(x, device):
    return torch.min(x, torch.tensor([30.0], device = device)).exp()


def sparse_max(x, mask, device):
    """对一个稀疏矩阵求最大值(精确)"""

    if device.type == 'cuda':
        dense_mat = torch.cuda.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(),
                                                  device = device).to_dense()
    else:
        dense_mat = torch.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device = device).to_dense()

    return torch.max(dense_mat, 0)[0] + x.min() - 1


def sparse_argmax(x, mask, device):
    """对于属于的行向量, 求准确的最大值所在的位置"""

    if device.type == 'cuda':
        dense_mat = torch.cuda.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(),
                                                  device = device).to_dense()
    else:
        dense_mat = torch.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device = device).to_dense()

    return torch.argmax(dense_mat, 0)

