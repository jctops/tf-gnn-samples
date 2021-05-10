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


MODEL_TYPES = ["GGNN", "RGCN", "RGAT", "RGIN"]#, "GNN-Edge-MLP0", "GNN-Edge-MLP1", "GNN_FiLM"]
TASKS = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]
USE_SDRF = [True, False]

pbs_array_index = int(os.environ['PBS_ARRAY_INDEX'])
model, task, use_sdrf = list(product(MODEL_TYPES, TASKS, USE_SDRF))[pbs_array_index]
MODEL_TYPES = [model]
TASKS = [task]
USE_SDRF = [use_sdrf]

TEST_RES_RE = re.compile('^Metrics: MAEs: \d+:([0-9.]+) \| Error Ratios: \d+:([0-9.]+)')
TIME_RE = re.compile('^Training took (\d+)s')


def run(args):
    target_dir = args['LOG_TARGET_DIR']
    os.makedirs(target_dir, exist_ok=True)
    print("Starting QM9 experiments, will write logfiles for runs into %s." % target_dir)
    num_seeds = int(args.get('--num-runs'))
    results = {}
    for use_sdrf in USE_SDRF:
        results[use_sdrf] = {}
        for model in MODEL_TYPES:
            results[use_sdrf][model] = [{"test_errors": [], "times": []} for _ in TASKS]
            for task_id in range(len(TASKS)):
                for seed in range(1, 1 + num_seeds):
                    logfile = os.path.join(target_dir, "%s_task%i_seed%i_sdrf%s.txt" % (model, task_id, seed, use_sdrf))
                    with open(logfile, "w") as log_fh:
                        subprocess.check_output(["python",
                                            "train.py",
                                            "--run-test",
                                            model,
                                            "QM9",
                                            "--model-param-overrides",
                                            "{\"random_seed\": %i}" % seed,
                                            "--task-param-overrides",
                                            "{\"task_ids\": [%i], \"preprocess_with_sdrf\": \"%s\"}" % (task_id, use_sdrf),
                                            ],
                                            stdout=log_fh,
                                            stderr=log_fh)
                    with open(logfile, "r") as log_fh:
                        for line in log_fh.readlines():
                            time_match = TIME_RE.search(line)
                            res_match = TEST_RES_RE.search(line)
                            if time_match is not None:
                                results[use_sdrf][model][task_id]["times"].append(int(time_match.groups()[0]))
                            elif res_match is not None:
                                results[use_sdrf][model][task_id]["test_errors"].append(float(res_match.groups()[1]))

    row_fmt_string = "%7s " + "&% 35s " * len(MODEL_TYPES) + "\\\\"
    print(row_fmt_string % tuple([""] + MODEL_TYPES))
    for task_id, task in enumerate(TASKS):
        model_results = []
        for use_sdrf in USE_SDRF:
            for model in MODEL_TYPES:
                err = np.mean(results[use_sdrf][model][task_id]["test_errors"])
                std = np.std(results[use_sdrf][model][task_id]["test_errors"])
                time_in_min = np.mean(results[use_sdrf][model][task_id]["times"]) / 60
                model_results.append("%.2f & ($\pm %.2f$; $%.1f$min)" % (err, std, time_in_min))
        print(row_fmt_string % tuple([task] + model_results))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])
