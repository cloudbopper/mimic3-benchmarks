"""Generate pruned datasets and train models"""
import argparse
import os
import shutil
import subprocess
import tarfile

import numpy as np


PREPROCESS = "preprocess"
TRAIN = "train"
EVALUATE = "evaluate"


def main():
    """Identify model state with lowest validation set loss"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", required=True, help="Directory containing original data set in requisite folder structure (small part or all data)")
    parser.add_argument("-features_filename", required=True, help="Features cloudpickle file that provides that pruning information")
    parser.add_argument("-start_seed", type=int, default=1284171779)
    parser.add_argument("-num_datasets", type=int, default=2)
    parser.add_argument("-modes", choices=[PREPROCESS, TRAIN, EVALUATE], nargs="+", required=True)
    args = parser.parse_args()
    return pipeline(args)


def pipeline(args):
    """Pipeline"""
    cwd = os.getcwd()
    run_dirs = [f"run_dir_{seed}" for seed in range(args.start_seed, args.start_seed + args.num_datasets)]
    if PREPROCESS in args.modes:
        condor_dir = f"{os.path.dirname(os.path.realpath(__file__))}/condor"
        for idx, run_dir in enumerate(run_dirs):
            # Make directory
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            pruned_dir = f"{run_dir}/pruned"
            seed = args.start_seed + idx
            # Prune data
            cmd = (f"python -m mimic3ext.prune_data -data_dir {args.data_dir} -features_filename {args.features_filename}"
                   f" -output_dir {pruned_dir} -randomize_features -seed {seed}")
            subprocess.check_call(cmd, shell=True)
            # Compress pruned data
            with tarfile.open(f"{run_dir}/pruned.tar.gz", "w:gz") as tar_fp:
                os.chdir(pruned_dir)
                for filename in os.listdir("."):
                    tar_fp.add(filename)
                os.chdir(cwd)
            # Delete raw data files
            shutil.rmtree(pruned_dir)
            # Copy other files
            for filename in os.listdir(condor_dir):
                shutil.copy(f"{condor_dir}/{filename}", run_dir)
    if TRAIN in args.modes:
        for run_dir in run_dirs:
            os.chdir(run_dir)
            cmd = "condor_submit mimic.sub"
            subprocess.check_call(cmd, shell=True)
            os.chdir(cwd)
    if EVALUATE in args.modes:
        last_aucs = np.zeros(args.num_datasets)
        best_aucs = np.zeros(args.num_datasets)
        for idx, run_dir in enumerate(run_dirs):
            pass  # TODO: parse outputs and identify AUCs


if __name__ == "__main__":
    main()
