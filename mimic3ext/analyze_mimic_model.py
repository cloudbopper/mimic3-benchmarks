"""Analyze MIMIC-III in-hospital-mortality model using anamod"""

import argparse
import glob
import os
import shutil
import subprocess

import cloudpickle
import anamod
from anamod import ModelAnalyzer
import numpy as np

from mimic3ext import ext_utils, model_loader, ModelWrapper

SAVED_ARCHIVE = f"{ext_utils.SAVED}.tar.gz"  # File with saved model architecture, data
SAVED_DIR = f"wdir/mimic3-benchmarks/mimic3models/in_hospital_mortality/{ext_utils.SAVED}/"
STATES_ARCHIVE = "saved_model_states.tar.gz"
STATES_GLOB = "wdir/mimic3-benchmarks/mimic3models/in_hospital_mortality/keras_states/*epoch100.*"  # To locate extracted state file


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", required=True)
    parser.add_argument("-output_dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = anamod.utils.get_logger(__name__, f"{args.output_dir}/analyze_mimic.log")

    logger.info("Extracting model architecture and state archives")
    saved_filename = f"{args.input_dir}/{SAVED_ARCHIVE}"
    if not os.path.exists(SAVED_ARCHIVE.split(".")[0]):
        subprocess.check_call(f"tar --directory {args.input_dir} -xvzf {saved_filename}", shell=True)
    states_filename = f"{args.input_dir}/{STATES_ARCHIVE}"
    if not os.path.exists(STATES_ARCHIVE.split(".")[0]):
        subprocess.check_call(f"tar --directory {args.input_dir} -xvzf {states_filename}", shell=True)

    logger.info("Loading TF model and weights")
    filenames = glob.glob(f"{args.input_dir}/{STATES_GLOB}")
    assert len(filenames) == 1
    tf_model = model_loader.build_model(filenames[0])
    # Create model wrapper
    model = ModelWrapper(tf_model)

    logger.info("Loading data")
    np_data = np.load(f"{args.input_dir}/{SAVED_DIR}/{ext_utils.VAL}_{ext_utils.DATA_FILENAME}")
    data = np_data["arr_0"]
    data = np.transpose(data, (0, 2, 1))  # to get instances X features X timestamps
    targets = np_data["arr_1"]

    # Copy file containing feature names
    shutil.copy(f"{args.input_dir}/{SAVED_DIR}/{ext_utils.DISCRETIZER_HEADER_FILENAME}", args.output_dir)

    logger.info("Analyzing model")
    analyzer = ModelAnalyzer(model, data, targets,
                             output_dir=args.output_dir, model_loader_filename=os.path.abspath(model_loader.__file__),
                             condor=True, shared_filesystem=True, features_per_worker=1, cleanup=False,
                             window_search_algorithm="effect_size", retry_arbitrary_failures=1,
                             num_shuffling_trials=200,
                             window_effect_size_threshold=0.05)
    features = analyzer.analyze()
    with open(f"{args.output_dir}/features.cpkl", "wb") as features_file:
        cloudpickle.dump(features, features_file)


if __name__ == "__main__":
    main()

