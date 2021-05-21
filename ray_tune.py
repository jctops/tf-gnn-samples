"""
distributed hyperparameter tuning
"""
import argparse
import os
import time
from functools import partial

import numpy as np
import torch

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from train import run
from test import test
from models import ggnn_model, rgat_model, rgcn_model, rgin_model
from tasks.qm9_task import QM9_Task
from utils.model_utils import name_to_task_class, name_to_model_class

def average_test(models, datas):

  results = [test(model, data) for model, data in zip(models, datas)]
  train_accs, val_accs, tmp_test_accs = [], [], []

  for train_acc, val_acc, test_acc in results:
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    tmp_test_accs.append(test_acc)

  return train_accs, val_accs, tmp_test_accs

def set_search_space(opt):
  if opt["dataset"] == "qm9":
    return set_qm9_search_space(opt)

def set_qm9_search_space(opt):
  """
  set the search space for hyperparams here
  """
  opt["decay"] = tune.loguniform(2e-3, 1e-2)
  # opt['curvature_fn'] = tune.choice(['uma_forman'])
  opt['sdrf_curvature_fn'] = tune.choice(['unbiased_forman'])
  opt['sdrf_consider_positivity'] = tune.choice([True, False])
  opt['sdrf_target_curvature'] = tune.uniform([-0.5, 0.5])
  opt['sdrf_scaling'] = tune.loguniform([5,100])
  # 'max_steps':[2,3,5,10]
  opt['sdrf_max_steps'] = tune.uniform([0.1, 0.5])



def train_ray(opt, checkpoint_dir=None, data_dir="../data"):
  # get data
  task_cls, _ = name_to_task_class('qm9')
  # get model
  model_cls, additional_model_params = name_to_model_class(opt['MODEL_NAME'])

  # Collect parameters from first the class defaults, potential task defaults, and then CLI:
  task_params = task_cls.default_params()
  model_params = model_cls.default_params()
  model_params.update(additional_model_params)


  for split in range(opt["num_init"]):





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
    task_params.update(json.loads(args.get('--task-param-overrides') or '{}'))
    model_params.update(json.loads(args.get('--model-param-overrides') or '{}'))

    # Now prepare to actually run by setting up directories, creating object instances and running:
    result_dir = args.get('--result-dir', 'trained_models')
    os.makedirs(result_dir, exist_ok=True)
    task = task_cls(task_params)  # task params needs to contain all flow args
    data_path = opt.get('--data-path') or task.default_data_path()
    data_path = RichPath.create(data_path, azure_info_path)
    task.load_data(data_path)

    model.initialize_model()
    model.train(quiet=opt.get('--quiet'), tf_summary_path=opt.get('--tensorboard'))

    train_this = train

    models.append(model)


    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
      checkpoint = os.path.join(checkpoint_dir, "checkpoint")
      model_state, optimizer_state = torch.load(checkpoint)
      model.load_state_dict(model_state)
      optimizer.load_state_dict(optimizer_state)

  for epoch in range(1, opt["epoch"]):
    train_loss, train_task_metrics, train_num_graphs, train_graphs_p_s, train_nodes_p_s, train_edges_p_s = \
      model.__run_epoch("epoch %i (training)" % epoch,
                       model.task._loaded_data[DataFold.TRAIN],
                       DataFold.TRAIN,
                       quiet=quiet,
                       summary_writer=train_writer)
    train_accs, val_accs, tmp_test_accs = average_test(models, datas)

    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      best = np.argmax(val_accs)
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
    tune.report(loss=train_loss, accuracy=train_task_metrics[f"abs_err_task{opt['task_id']}"])

    # tune.report(loss=train_loss, accuracy=np.mean(val_accs), test_acc=np.mean(tmp_test_accs), train_acc=np.mean(train_accs))


def main(opt):
  data_dir = os.path.abspath("../data")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  opt = set_search_space(opt)
  scheduler = ASHAScheduler(
    metric=opt['metric'],
    mode="max",
    max_t=opt["epoch"],
    grace_period=opt["grace_period"],
    reduction_factor=opt["reduction_factor"],
  )
  reporter = CLIReporter(
    metric_columns=["accuracy", "test_acc", "train_acc", "loss", "training_iteration"]
  )

  train_fn = train_ray

  result = tune.run(
    partial(train_fn, data_dir=data_dir),
    name=opt["name"],
    resources_per_trial={"cpu": opt["cpus"], "gpu": opt["gpus"]},
    search_alg=None,
    keep_checkpoints_num=3,
    checkpoint_score_attr=opt['metric'],
    config=opt,
    num_samples=opt["num_samples"],
    scheduler=scheduler,
    max_failures=2,
    local_dir="../ray_tune",
    progress_reporter=reporter,
    raise_on_failed_trial=False,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dataset", type=str, default="qm9", help="qm9, Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
  )
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
