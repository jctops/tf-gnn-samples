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


#MODEL_TYPES = ["GGNN", "RGCN", "RGAT", "RGIN"]#, "GNN-Edge-MLP0", "GNN-Edge-MLP1", "GNN_FiLM"]
MODEL_TYPES = ["GGNN", "RGCN"]
#MODEL_TYPES = ["RGAT", "RGIN"]
TASKS = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "Omega"]
USE_SDRF = [True, False]
#USE_SDRF = [False]
#USE_SDRF = [True]

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
                    if os.path.isfile(logfile):
                        with open(logfile, "r") as log_fh:
                            for line in log_fh.readlines():
                                time_match = TIME_RE.search(line)
                                res_match = TEST_RES_RE.search(line)
                                if time_match is not None:
                                    results[use_sdrf][model][task_id]["times"].append(int(time_match.groups()[0]))
                                elif res_match is not None:
                                    results[use_sdrf][model][task_id]["test_errors"].append(float(res_match.groups()[1]))

    row_fmt_string = "%7s " + "&% 35s " * len(MODEL_TYPES) * len(USE_SDRF) + "\\\\"
    print(row_fmt_string % tuple([""] + [f'{model}_sdrf{use_sdrf}' for model in MODEL_TYPES for use_sdrf in USE_SDRF]))
    for task_id, task in enumerate(TASKS):
        model_results = []
        for model in MODEL_TYPES:
            for use_sdrf in USE_SDRF:
                #print(f'{model}_sdrf{use_sdrf}')
                #print(results[use_sdrf][model][task_id])
                err = np.mean(results[use_sdrf][model][task_id]["test_errors"])
                std = np.std(results[use_sdrf][model][task_id]["test_errors"])
                time_in_min = np.mean(results[use_sdrf][model][task_id]["times"]) / 60
                model_results.append("%.2f ($\pm %.2f$)" % (err, std))
        print(row_fmt_string % tuple([task] + model_results))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])
