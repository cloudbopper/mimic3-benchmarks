"""Prune MIMIC model data according to analyzed features"""

import argparse
import os

import cloudpickle
import numpy as np

import anamod
from mimic3ext import ext_utils

SAVED_DIR = f"wdir/mimic3-benchmarks/mimic3models/in_hospital_mortality/{ext_utils.SAVED}/"


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", required=True)
    parser.add_argument("-features_filename", required=True)
    parser.add_argument("-output_dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = anamod.utils.get_logger(__name__, f"{args.output_dir}/prune_data.log")

    logger.info("Loading features")
    with open(args.features_filename, "rb") as features_file:
        features = cloudpickle.load(features_file)

    logger.info("Writing important features")
    with open(f"{args.output_dir}/features.cpkl", "wb") as features_file:
        cloudpickle.dump(features, features_file)

    logger.info("Loading data")
    train_set = np.load(f"{args.data_dir}/{SAVED_DIR}/{ext_utils.TRAIN}_{ext_utils.DATA_FILENAME}")
    train_data = prune_data(train_set["arr_0"], features)
    test_set = np.load(f"{args.data_dir}/{SAVED_DIR}/{ext_utils.TEST}_{ext_utils.DATA_FILENAME}")
    test_data = prune_data(test_set["arr_0"], features)
    val_set = np.load(f"{args.data_dir}/{SAVED_DIR}/{ext_utils.VAL}_{ext_utils.DATA_FILENAME}")
    val_data = prune_data(val_set["arr_0"], features)

    logger.info("Writing data")
    np.savez(f"{args.output_dir}/{ext_utils.TRAIN}_{ext_utils.DATA_FILENAME}", train_data, train_set["arr_1"])
    np.savez(f"{args.output_dir}/{ext_utils.TEST}_{ext_utils.DATA_FILENAME}", test_data, test_set["arr_1"])
    np.savez(f"{args.output_dir}/{ext_utils.VAL}_{ext_utils.DATA_FILENAME}", val_data, val_set["arr_1"])


def prune_data(data, features):
    """Prune data according to feature importance analysis"""
    # data: instances X timesteps X features
    num_features = len(features)
    assert data.shape[2] == num_features
    fids = [idx for idx in range(num_features) if features[idx].important]
    new_data = data[:, :, fids]
    for idx, fid in enumerate(fids):
        window = features[fid].temporal_window
        assert window is not None
        left, right = window
        new_data[:, 0: left, idx] = 0
        new_data[:, right + 1:, idx] = 0
    return new_data


if __name__ == "__main__":
    main()
