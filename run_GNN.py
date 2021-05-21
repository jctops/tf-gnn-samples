"""
distributed hyperparameter tuning
"""
import argparse
import os
import json
import time
from functools import partial
from dpu_utils.utils import RichPath
import numpy as np

from train import run
from test import test
from models import ggnn_model, rgat_model, rgcn_model, rgin_model
from tasks.qm9_task import QM9_Task
from tasks import DataFold
from utils.model_utils import name_to_task_class, name_to_model_class

MODEL_TYPES = ["GGNN", "RGCN", "RGAT", "RGIN"]
TASKS = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]


def setup(opt, data_path='data/qm9'):
  azure_info_path = opt.get('--azure-info', None)
  # get model and task
  model_cls, additional_model_params = name_to_model_class(opt['model_name'])
  task_cls, additional_task_params = name_to_task_class(opt['task_name'])

  # Collect parameters from first the class defaults, potential task defaults, and then CLI:
  task_params = task_cls.default_params()
  task_params.update(additional_task_params)
  model_params = model_cls.default_params()
  model_params.update(additional_model_params)

  # Load potential task-specific defaults:
  task_model_default_hypers_file = \
    os.path.join(os.path.dirname(__file__),
                 "tasks",
                 "default_hypers",
                 "%s_%s.json" % (task_cls.name(), model_cls.name(model_params)))
  if os.path.exists(task_model_default_hypers_file):
    print("Loading task/model-specific default parameters from %s." % task_model_default_hypers_file)
    with open(task_model_default_hypers_file, "rt") as f:
      default_task_model_hypers = json.load(f)
    task_params.update(default_task_model_hypers['task_params'])
    model_params.update(default_task_model_hypers['model_params'])

  # Load overrides from command line:
  task_params.update(json.loads(opt.get('--task-param-overrides') or '{}'))
  task_params['preprocess_with_sdrf'] = False
  print(f"task parameters: {task_params}")
  model_params.update(json.loads(opt.get('--model-param-overrides') or '{}'))

  # Finally, upgrade every parameters that's a path to a RichPath:
  task_params_orig = dict(task_params)
  for (param_name, param_value) in task_params.items():
    if param_name.endswith("_path"):
      task_params[param_name] = RichPath.create(param_value, azure_info_path)

  # Now prepare to actually run by setting up directories, creating object instances and running:
  result_dir = opt.get('--result-dir', 'trained_models')
  os.makedirs(result_dir, exist_ok=True)
  task = task_cls(task_params)

  # load the data
  data_path = RichPath.create(data_path, azure_info_path)
  task.load_data(data_path)

  # configure the model
  run_id = "_".join(
    [task_cls.name(), model_cls.name(model_params), time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
  model = model_cls(model_params, task, run_id, result_dir)
  model.log_line("Run %s starting." % run_id)
  model.log_line(" Using the following task params: %s" % json.dumps(task_params_orig))
  model.log_line(" Using the following model params: %s" % json.dumps(model_params))
  model.initialize_model()
  print(f"model dictionary {dir(model)}")
  return model


def main(opt):

  model = setup(opt)
  for epoch in range(1, opt["epoch"]):
    train_loss, train_task_metrics, train_num_graphs, train_graphs_p_s, train_nodes_p_s, train_edges_p_s = \
      model._Sparse_Graph_Model__run_epoch("epoch %i (training)" % epoch,
                        model.task._loaded_data[DataFold.TRAIN],
                        DataFold.TRAIN, quiet=False)

    print("\r\x1b[K", end='')
    model.log_line(" Train: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | edges/sec: %.0f"
                % (train_loss,
                   model.task.pretty_print_epoch_task_metrics(train_task_metrics, train_num_graphs),
                   train_graphs_p_s, train_nodes_p_s, train_edges_p_s))

    valid_loss, valid_task_metrics, valid_num_graphs, valid_graphs_p_s, valid_nodes_p_s, valid_edges_p_s = \
      model._Sparse_Graph_Model__run_epoch("epoch %i (validation)" % epoch,
                       model.task._loaded_data[DataFold.VALIDATION],
                       DataFold.VALIDATION,
                       quiet=False)
    print("\r\x1b[K", end='')
    valid_metric_descr = \
      model.task.pretty_print_epoch_task_metrics(valid_task_metrics, valid_num_graphs)
    model.log_line(" Valid: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | edges/sec: %.0f"
                  % (valid_loss, valid_metric_descr, valid_graphs_p_s, valid_nodes_p_s, valid_edges_p_s))

    # data = model.task._loaded_data.get(DataFold.TEST)
    # if data is None:
    #   data = model.task.load_eval_data_from_path(test_data_path)
    # test_loss, test_task_metrics, test_num_graphs, _, _, _ = \
    #   model._Sparse_Graph_Model__run_epoch("Test", data, DataFold.TEST, quiet=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dataset", type=str, default="qm9", help="qm9, Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
  )
  parser.add_argument(
    "--model_name", type=str, default="ggnn", help=f"choices are {MODEL_TYPES}"
  )
  parser.add_argument(
    "--task_name", type=str, default="qm9", help=f"choices are {TASKS}"
  )
  parser.add_argument("--epoch", type=int, default=20, help="number of epochs")
  # ray args
  parser.add_argument("--num_samples", type=int, default=20, help="number of ray trials")
  parser.add_argument("--gpus", type=float, default=0, help="number of gpus per trial. Can be fractional")
  parser.add_argument("--cpus", type=float, default=1, help="number of cpus per trial. Can be fractional")
  parser.add_argument(
    "--grace_period", type=int, default=10, help="number of epochs to wait before terminating trials"
  )
  parser.add_argument(
    "--reduction_factor", type=int, default=10, help="number of trials is halved after this many epochs"
  )
  parser.add_argument("--name", type=str, default="ray_exp")
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random splits >= 0. 0 for planetoid split")
  parser.add_argument("--num_init", type=int, default=1, help="Number of random initializations >= 0")

  parser.add_argument('--metric', type=str, default='accuracy', help='metric to sort the hyperparameter tuning runs on')
  args = parser.parse_args()

  opt = vars(args)

  main(opt)
