import multiprocessing
import torch
import torch.nn as nn
import os
import time

import numpy as np
import model as SPModel
import util
from aggregators import SurveyAggregator, SurveyNeuralPredictor


def _module(model):
    """如果可以并行化, 则并行处理"""
    return model.module if isinstance(model, nn.DataParallel) else model


class PropagatorDecimatorSolver(nn.Module):
    """SP-NUERAL Solver 的基类"""

    def __init__(self, device, name, feature_dim, hidden_dimension,
                 agg_hidden_dimension = 100, func_hidden_dimension = 100, agg_func_dimension = 50,
                 classifier_dimension = 50, local_search_iterations = 1000, pure_sp = True):

        super(PropagatorDecimatorSolver, self).__init__()
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
        # self._cnf_evaluator = solver.SatCNFEvaluator(device)
        # self._loss_evaluator = solver.SatLossEvaluator(alpha = alpha, device = self._device)

        self._module_list.append(self._propagator)
        self._module_list.append(self._predictor)

        self._global_step = nn.Parameter(torch.tensor([0], dtype = torch.float, device = self._device),
                                         requires_grad = False)
        # self._temperature = nn.Parameter(torch.tensor([temperature], dtype = torch.float, device = self._device),
        #                                  requires_grad = False)
        self._name = name
        self._local_search_iterations = local_search_iterations
        self._eps = 1e-8 * torch.ones(1, device = self._device, requires_grad = False)
        '''是否使用神经网络的 decimate 方法'''
        self._pure_sp = pure_sp

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load(self, import_path_base):
        self.load_state_dict(torch.load(os.path.join(import_path_base, self._name)))

    def save(self, export_path_base):
        """保存模型"""
        torch.save(self.state_dict(), os.path.join(export_path_base, self._name))

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
            function_prediction = (flag * prediction[1]).view(sat_problem._batch_replication, -1).sum(dim = 0) \
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

    # def _check_recurrence_termination(self, active, prediction, sat_problem):
    #     """停用模型已找到SAT解决方案的CNF"""
    #
    #     _, res, _ = self._cnf_evaluator(variable_prediction = prediction[0],
    #                                                  graph_map = sat_problem._graph_map,
    #                                                  batch_variable_map = sat_problem._batch_variable_map,
    #                                                  batch_function_map = sat_problem._batch_function_map,
    #                                                  edge_feature = sat_problem._edge_feature,
    #                                                  meta_data = sat_problem._meta_data, is_training = False)
    #     # .detach().cpu().numpy()
    #
    #     if sat_problem._batch_replication > 1:
    #         real_batch = torch.mm(sat_problem._replication_mask_tuple[1], (res[0] > 0.5).float())
    #         dup_batch = torch.mm(sat_problem._replication_mask_tuple[0], (real_batch == 0).float())
    #         active[active[:, 0], 0] = (dup_batch[active[:, 0], 0] > 0)
    #     else:
    #         active[active[:, 0], 0] = (res[0][active[:, 0], 0] <= 0.5)

    # def _compute_evaluation_metrics(self, model, evaluator, prediction, label, sat_problem):
    #     return evaluator(model, prediction, label, sat_problem)

    # def _compute_evaluation_metrics(self, prediction, label, graph_map, batch_variable_map,
    #                                 batch_function_map, edge_feature, meta_data, vf_mask, sat_problem = None):
    #     """交叉验证"""
    #     _, res, _ = self._cnf_evaluator(variable_prediction = prediction, graph_map = graph_map,
    #                                                  batch_variable_map = batch_variable_map,
    #                                                  batch_function_map = batch_function_map,
    #                                                  edge_feature = edge_feature, vf_mask = vf_mask,
    #                                                  sat_problem = sat_problem, is_training = False)
    #     # self._temperature = temperature
    #     recall = torch.sum(label * ((res[0] > 0.5).float() - label).abs()) / torch.max(torch.sum(label), self._eps)
    #     accuracy = nn.L1Loss((res[0] > 0.5).float(), label).unsqueeze(0)
    #     loss_value = self._loss_evaluator(variable_prediction = prediction, label = label, graph_map = graph_map,
    #                                       batch_variable_map = batch_variable_map,
    #                                       batch_function_map = batch_function_map,
    #                                       edge_feature = edge_feature, meta_data = meta_data,
    #                                       global_step = self._global_step, eps = self._eps).unsqueeze(0)
    #     return torch.cat([accuracy, recall, loss_value], 0)

    # def compute_loss(self, model, loss, mode, prediction, label, sat_problem = None):
    #     return loss(prediction, label)

    # def compute_loss(self, mode, prediction, label, sat_problem = None):
    #     if mode:
    #         '''训练集'''
    #         res = self._loss_evaluator(sat_problem._graph_map,sat_problem._batch_variable_map,
    #                                    sat_problem._batch_function_map, sat_problem._edge_feature, sat_problem._meta_data,
    #                                    global_step = self._global_step, eps = self._eps)
    #
    #         return res
    #     else:
    #         '''验证集'''
    #         res = self._compute_evaluation_metrics(prediction, label, sat_problem)
    #         return res.cpu().numpy()

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
    def __init__(self, device, use_cuda, dimacs_file, validate_file = None, epoch_replication = 3, batch_replication = 1,
                 epoch = 100, batch_size = 2000, hidden_dimension = 1, feature_dim = 100, train_outer_recurrence_num = 1,
                 error_dim = 3):
        self._use_cuda = use_cuda
        '''读入数据路径'''
        self._dimacs_file = dimacs_file
        self._validate_file = validate_file
        self._epoch_replication = epoch_replication
        self._batch_replication = batch_replication
        self._epoch = epoch
        self._batch_size = batch_size
        self._hidden_dimension = 2 * hidden_dimension
        self._feature_dim = feature_dim
        '''以下 limit 设置是用于控制 data_loader 一次读入多少行数据'''
        self._train_batch_limit = 20000
        self._test_batch_limit = 40000
        self._max_cache_size = 100000
        self._train_outer_recurrence_num = train_outer_recurrence_num
        self._device = device
        self._num_cores = multiprocessing.cpu_count()
        self._error_dim = error_dim
        torch.set_num_threads(self._num_cores)
        '''初始化模型列表'''
        model_list = [PropagatorDecimatorSolver(self._device, "sp-nueral-solver", self._feature_dim, hidden_dimension)]
        '''将模型放到 GPU 上运行 如果 cuda 设备可用'''
        self._model_list = [self._set_device(model) for model in model_list]

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

    def _reset_global_step(self):
        """重新计数训练轮数"""
        for model in self._model_list:
            _module(model)._global_step.data = torch.tensor([0], dtype = torch.float, device = self._device)

    def _compute_evaluation_metrics(self, model, evaluator, prediction, label, sat_problem):
        return evaluator(model, prediction, label, sat_problem)

    def _compute_loss(self, model, loss, mode, prediction, label, sat_problem = None):
        return loss(prediction, label)

    def _test_epoch(self, validation_loader, is_train = False, batch_replication = 1):
        """用于做模型的验证"""
        with torch.no_grad():
            error = np.zeros((self._error_dim, len(self._model_list)), dtype = np.float32)
            errors = SPModel.test_batch(self, validation_loader, error, self._model_list, self._device, batch_replication,
                                        self._use_cuda, is_train, False)
            return errors

    def train(self, last_export_path_base = None, best_export_path_base = None, metric_index = 0, load_model = None,
              train_epoch_size = 0, reset_step = False, is_train = True, parallel = False):
        """训练集"""
        train_loader = util.FactorGraphDataset.get_loader(
            input_file = self._dimacs_file, limit = self._train_batch_limit,
            hidden_dim = self._hidden_dimension, batch_size = self._batch_size, shuffle = True,
            num_workers = self._num_cores, max_cache_size = self._max_cache_size,
            epoch_size = train_epoch_size, parallel = parallel)
        '''验证集'''
        validation_loader = util.FactorGraphDataset.get_loader(
            input_file = self._validate_file, limit = self._train_batch_limit,
            hidden_dim = self._hidden_dimension, batch_size = self._batch_size, shuffle = False,
            num_workers = self._num_cores, max_cache_size = self._max_cache_size)

        '''设置模型保存路径 若不存在 则创建该文件夹'''
        if not os.path.exists(best_export_path_base):
            os.makedirs(best_export_path_base)

        if not os.path.exists(last_export_path_base):
            os.makedirs(last_export_path_base)
        losses = np.zeros((len(self._model_list), self._epoch, self._epoch_replication), dtype = np.float32)
        errors = np.zeros((self._error_dim, len(self._model_list), self._epoch, self._epoch_replication),
                          dtype = np.float32)
        best_errors = np.repeat(np.inf, len(self._model_list))

        for rep in range(self._epoch_replication):
            '''在之前模型的基础上进行训练'''
            if load_model == "best" and best_export_path_base is not None:
                self._load(best_export_path_base)
            elif load_model == "last" and last_export_path_base is not None:
                self._load(last_export_path_base)
            if reset_step:
                self._reset_global_step()
            for epoch in range(self._epoch):
                start_time = time.time()
                total_loss = np.zeros(len(self._model_list), dtype = np.float32)
                losses[:, epoch, rep] = SPModel.train_batch(self, train_loader, total_loss, rep, epoch, self._model_list,
                                                            self._device, self._batch_replication,
                                                            self._hidden_dimension, self._feature_dim,
                                                            self._train_outer_recurrence_num, self._use_cuda, is_train,
                                                            True)

                if self._use_cuda:
                    torch.cuda.empty_cache()
                '''验证过程'''
                errors[:, :, epoch, rep] = self._test_epoch(validation_loader, not is_train, 1)
                duration = time.time() - start_time
                if last_export_path_base is not None:
                    '''存储最后一次生成的模型'''
                    for (i, model) in enumerate(self._model_list):
                        _module(model).save(last_export_path_base)
                if best_export_path_base is not None:
                    '''存储最佳的模型'''
                    for (i, model) in enumerate(self._model_list):
                        if errors[metric_index, i, epoch, rep] < best_errors[i]:
                            best_errors[i] = errors[metric_index, i, epoch, rep]
                            _module(model).save(best_export_path_base)
                if self._use_cuda:
                    torch.cuda.empty_cache()
                message = ''
                for (i, model) in enumerate(self._model_list):
                    name = _module(model)._name
                    message += 'Step {:d}: {:s} error={:s}, loss={:5.5f} |'. \
                        format(_module(model)._global_step.int()[0],
                               name, np.array_str(errors[:, i, epoch, rep].flatten()),
                               losses[i, epoch, rep])

                print('Rep {:2d}, Epoch {:2d}: {:s}'.format(rep + 1, epoch + 1, message))
                print('Time spent: %s seconds' % duration)
        if self._use_cuda:
            torch.backends.cudnn.benchmark = False
        if best_export_path_base is not None:
            '''存储 loss 和 errors'''
            base = os.path.relpath(best_export_path_base)
            np.save(os.path.join(base, "losses"), losses, allow_pickle = False)
            np.save(os.path.join(base, "errors"), errors, allow_pickle = False)
            '''保存模型'''
            self._save(best_export_path_base)

    def predict(self, import_path_base = None, epoch_replication = 1, randomized = False):
        """模型的预测功能"""
        with torch.no_grad():
            test_loader = util.FactorGraphDataset.get_loader(
                input_file = self._dimacs_file, limit = self._test_batch_limit,
                hidden_dim = self._hidden_dimension, batch_size = self._batch_size, shuffle = False,
                num_workers = self._num_cores, max_cache_size = self._max_cache_size,
                batch_replication = epoch_replication)
            '''加载模型'''
            if import_path_base is not None:
                self._load(import_path_base)

            SPModel.predict_batch(test_loader, self._model_list, self._device, epoch_replication, self._use_cuda,
                                  randomized = randomized)
