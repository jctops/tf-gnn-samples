"""
distributed hyperparameter tuning
"""
import argparse
import os
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from tasks import DataFold

from test import test
from run_GNN import setup, MODEL_TYPES, TASKS
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
  opt['sdrf_target_curvature'] = tune.uniform(-0.5, 0.5)
  opt['sdrf_scaling'] = tune.loguniform(5, 100)
  # 'max_steps':[2,3,5,10]
  opt['sdrf_max_steps'] = tune.uniform(0.1, 0.5)
  return opt


def get_aggregate_metric(task_metric_results, task_ids, num_graphs):
  fnum_graphs = float(num_graphs)
  maes = {}
  for task_id in task_ids:
    maes['mae_task%i' % task_id] = 0.
  for batch_task_metric_results in task_metric_results:
    for task_id in task_ids:
      maes['mae_task%i' % task_id] += batch_task_metric_results['abs_err_task%i' % task_id] / fnum_graphs
  # todo I think we just do one task at a time, but flagging in case I'm wrong
  print(f"mean average errors {maes}")
  return list(maes.values())[0]


def train_ray(opt, checkpoint_dir=None, data_dir="data/qm9"):
  model = setup(opt, data_dir)
  # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
  # should be restored.
  # if checkpoint_dir:
  #   checkpoint = os.path.join(checkpoint_dir, "checkpoint")
  #   model_state, optimizer_state = torch.load(checkpoint)
  #   model.load_state_dict(model_state)
  #   optimizer.load_state_dict(optimizer_state)
  task_ids = model.task.params['task_ids']

  for epoch in range(1, opt["epoch"]):
    train_loss, train_task_metrics, train_num_graphs, train_graphs_p_s, train_nodes_p_s, train_edges_p_s = \
      model._Sparse_Graph_Model__run_epoch("epoch %i (training)" % epoch,
                                           model.task._loaded_data[DataFold.TRAIN],
                                           DataFold.TRAIN, quiet=False)

    train_metric = get_aggregate_metric(train_task_metrics, task_ids, train_num_graphs)

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
    val_metric = get_aggregate_metric(valid_task_metrics, task_ids, train_num_graphs)

    tune.report(loss=train_loss, train_mae=train_metric, val_mae=val_metric, train_nps=train_nodes_p_s,
                val_nps=valid_nodes_p_s, train_gps=train_graphs_p_s, val_gps=valid_graphs_p_s)

    # tune.report(loss=train_loss, accuracy=np.mean(val_accs), test_acc=np.mean(tmp_test_accs), train_acc=np.mean(train_accs))


def main(opt):
  print(f"running with option {opt}")
  data_dir = os.path.abspath("data/qm9")
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
    max_failures=0,
    local_dir="ray_tune",
    progress_reporter=reporter,
    raise_on_failed_trial=False,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dataset", type=str, default="qm9", help="qm9, Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
  )
  parser.add_argument(
    "--model_name", type=str, default="ggnn", help=f"choices are {MODEL_TYPES}")
  parser.add_argument(
    "--task_name", type=str, default="qm9", help=f"choices are {TASKS}")
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
  parser.add_argument("--epoch", type=int, default=100, help="Max number of training epochs.")
  parser.add_argument("--name", type=str, default="ray_exp")
  parser.add_argument("--metric", type=str, default="val_mae", help='metric to sort the hyperparameter tuning runs on')
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random splits >= 0. 0 for planetoid split")
  parser.add_argument("--num_init", type=int, default=1, help="Number of random initializations >= 0")
  parser.add_argument("--preprocess_with_sdrf", action="store_true", help="Do the thing the paper is about")
  args = parser.parse_args()

  opt = vars(args)

  main(opt)
