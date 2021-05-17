#!/usr/bin/env python
"""
Usage:
    run_qm9_benchs.py [options] LOG_TARGET_DIR

Options:
    -h --help         Show this screen.
    --num-runs NUM    Number of runs to perform for each configuration. [default: 5]
    --debug           Turn on debugger.
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug
from itertools import product
import numpy as np
import os
import re
import subprocess

uma_args = {
    'curvature_fn':['uma_forman'],
    'consider_positivity':[True, False],
    'target_curvature':[-1,0,1],
    'scaling':[1,2,5,10],
    'max_steps':[2,3,5,10]
}
unbiased_args = {
    'curvature_fn':['unbiased_forman'],
    'consider_positivity':[True, False],
    'target_curvature':[-0.5,0,0.5],
    'scaling':[10,25,100],
    'max_steps':[2,3,5,10]
}
ARGUMENTS = list(product(*list(uma_args.values()))) + list(product(*list(unbiased_args.values())))


MODEL_TYPES = ["GGNN", "RGCN", "RGAT", "RGIN"]#, "GNN-Edge-MLP0", "GNN-Edge-MLP1", "GNN_FiLM"]
TASKS = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]

pbs_array_index = int(os.environ['PBS_ARRAY_INDEX'])
model_, task_, arguments_ = list(product(MODEL_TYPES, TASKS, ARGUMENTS))[pbs_array_index]
_, task_id_, _ = list(product(MODEL_TYPES, range(len(TASKS)), ARGUMENTS))[pbs_array_index]
MODEL_TYPES = [model_]
TASKS = [task_]
TASK_IDS = [task_id_]
ARGUMENTS = [arguments_]

TEST_RES_RE = re.compile('^Metrics: MAEs: \d+:([0-9.]+) \| Error Ratios: \d+:([0-9.]+)')
TIME_RE = re.compile('^Training took (\d+)s')


def run(args):
    target_dir = args['LOG_TARGET_DIR']
    os.makedirs(target_dir, exist_ok=True)
    print("Starting QM9 experiments, will write logfiles for runs into %s." % target_dir)
    num_seeds = int(args.get('--num-runs'))
    results = {}
    for arguments in ARGUMENTS:
        arguments_str = ';'.join([f'{x}={y}' for x,y in zip(list(uma_args.keys()), arguments)])
        curvature_fn, consider_positivity, target_curvature, scaling, max_steps = arguments
        results[arguments_str] = {}
        for model in MODEL_TYPES:
            results[arguments_str][model] = [{"test_errors": [], "times": []} for _ in TASKS]
            for task_id in TASK_IDS:
                for seed in range(1, 1 + num_seeds):
                    logfile = os.path.join(target_dir, "%s_task%i_seed%i_%s.txt" % (model, task_id, seed, arguments_str))
                    with open(logfile, "w") as log_fh:
                        subprocess.check_call(["python",
                                            "../train.py",
                                            "--run-test",
                                            model,
                                            "QM9",
                                            "--model-param-overrides",
                                            '{\"random_seed\": %i}' % seed,
                                            "--task-param-overrides",
                                            '''{\"task_ids\": [%i], \"preprocess_with_sdrf\": \"True\", \"sdrf_curvature_fn\": \"%s\",
\"sdrf_consider_positivity\": \"%s\", \"sdrf_target_curvature\": %f, \"sdrf_scaling\": %i, \"sdrf_max_steps\": %i}''' % \
(task_id, curvature_fn, consider_positivity, target_curvature, scaling, max_steps),
                                            ],
                                            stdout=log_fh,
                                            stderr=log_fh)
                    with open(logfile, "r") as log_fh:
                        for line in log_fh.readlines():
                            time_match = TIME_RE.search(line)
                            res_match = TEST_RES_RE.search(line)
                            if time_match is not None:
                                results[arguments_str][model][0]["times"].append(int(time_match.groups()[0]))
                            elif res_match is not None:
                                results[arguments_str][model][0]["test_errors"].append(float(res_match.groups()[1]))

    row_fmt_string = "%7s " + "&% 35s " * len(MODEL_TYPES) + "\\\\"
    print(row_fmt_string % tuple([""] + MODEL_TYPES))
    for task_id, task in enumerate(TASKS):
        task_id = 0 #task_id_
        model_results = []
        for arguments in ARGUMENTS:
            arguments_str = ';'.join([f'{x}={y}' for x,y in zip(list(uma_args.keys()), arguments)])
            for model in MODEL_TYPES:
                err = np.mean(results[arguments_str][model][task_id]["test_errors"])
                std = np.std(results[arguments_str][model][task_id]["test_errors"])
                time_in_min = np.mean(results[arguments_str][model][task_id]["times"]) / 60
                model_results.append("%.2f & ($\pm %.2f$; $%.1f$min)" % (err, std, time_in_min))
        print(row_fmt_string % tuple([task] + model_results))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])
