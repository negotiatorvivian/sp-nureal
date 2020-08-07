import argparse
import traceback

import torch
import torch.nn as nn
import solver
from base import SPNueralBase


def _module(model):
    return model.module if isinstance(model, nn.DataParallel) else model


class PropagatorDecimatorTrainer(SPNueralBase):
    def __init__(self, device, use_cuda, dimacs_file, validate_file, alpha = 0.1):
        super(PropagatorDecimatorTrainer, self).__init__(device, use_cuda, dimacs_file, validate_file)
        self._device = device
        self._cnf_evaluator = solver.SatCNFEvaluator(self._device)
        self._loss_evaluator = solver.SatLossEvaluator(alpha = alpha, device = self._device)

    def _check_recurrence_termination(self, active, prediction, sat_problem):
        """停用模型已找到SAT解决方案的CNF"""

        _, res, _ = self._cnf_evaluator(prediction[0], sat_problem._graph_map,
                                        sat_problem._batch_variable_map, sat_problem._batch_function_map,
                                        sat_problem._edge_feature, sat_problem._meta_data, is_training = False)
        # .detach().cpu().numpy()
        if sat_problem._batch_replication > 1:
            real_batch = torch.mm(sat_problem._replication_mask_tuple[1], (res[0] > 0.5).float())
            dup_batch = torch.mm(sat_problem._replication_mask_tuple[0], (real_batch == 0).float())
            active[active[:, 0], 0] = (dup_batch[active[:, 0], 0] > 0)
        else:
            active[active[:, 0], 0] = (res[0][active[:, 0], 0] <= 0.5)

    def _compute_evaluation_metrics(self, model, evaluator, prediction, label, sat_problem):
        """交叉验证"""
        _, res, _ = self._cnf_evaluator(prediction, sat_problem._graph_map, sat_problem._batch_variable_map,
                                        sat_problem._batch_function_map, sat_problem._edge_feature,
                                        sat_problem._vf_mask_tuple[3], sat_problem = sat_problem, is_training = False)
        # self._temperature = temperature
        recall = torch.sum(label * ((res[0] > 0.5).float() - label).abs()) / torch.max(torch.sum(label), model._eps)
        accuracy = nn.L1Loss()((res[0] > 0.5).float(), label).unsqueeze(0)
        loss_value = self._loss_evaluator(prediction, label, sat_problem._graph_map, sat_problem._batch_variable_map,
                                          sat_problem._batch_function_map, sat_problem._edge_feature,
                                          sat_problem._meta_data, global_step = model._global_step,
                                          eps = model._eps).unsqueeze(0)
        return torch.cat([accuracy, recall, loss_value], 0)

    def _compute_loss(self, model, loss, mode, prediction, label, sat_problem = None):
        if mode:
            '''训练集'''
            res = self._loss_evaluator(prediction, label, sat_problem._graph_map, sat_problem._batch_variable_map,
                                       sat_problem._batch_function_map, sat_problem._edge_feature,
                                       sat_problem._meta_data, global_step = model._global_step, eps = model._eps)

            return res
        else:
            '''验证集'''
            res = self._compute_evaluation_metrics(model, None, prediction, label, sat_problem)
            return res.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help = 'Path to datasets')
    parser.add_argument('-tv', '--validate', help = 'Path to validation datasets')
    parser.add_argument('-tl', '--last_model_path', help = 'Path to save the previous model')
    parser.add_argument('-tb', '--best_model_path', help = 'Path to save the best model')
    parser.add_argument('-g', '--gpu_mode', help = 'Run on GPU', action = 'store_true')
    parser.add_argument('-p', '--predict', help = 'Run prediction', action = 'store_true')
    parser.add_argument('-l', '--load_model', help = 'Load the previous model')
    parser.add_argument('-pl', '--load_model_path', help = 'Path to load the previous model')

    args = parser.parse_args()
    try:
        use_cuda = args.gpu_mode and torch.cuda.is_available()
        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        '''训练模式'''
        if not args.predict:
            '''检查必须的参数, 若缺失任意一个参数, 则抛出异常'''
            assert args.last_model_path and args.best_model_path and args.validate
            trainer = PropagatorDecimatorTrainer(device, use_cuda, args.dataset_path, args.validate)
            trainer.train(args.last_model_path, args.best_model_path)
        else:
            '''预测模式'''
            trainer = PropagatorDecimatorTrainer(device, use_cuda, args.dataset_path)
            '''检查必须的参数, 若缺失任意一个参数, 则抛出异常'''
            assert args.load_model_path
            trainer.predict(import_path_base = args.load_model_path)
    except:
        print(traceback.format_exc())
