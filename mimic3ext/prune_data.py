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
    parser.add_argument("-randomize_features", action="store_true", help="Instead of selecting features/timesteps listed in features file,"
                        " select a random set of features/timesteps of the same size")
    parser.add_argument("-seed", type=int, default=1241255801, help="RNG seed for randomizing features")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = anamod.utils.get_logger(__name__, f"{args.output_dir}/prune_data.log")

    logger.info("Loading features")
    with open(args.features_filename, "rb") as features_file:
        features = cloudpickle.load(features_file)

    logger.info("Selecting features")
    fids, windows = select_features(args, features)

    logger.info("Writing selected features")
    with open(f"{args.output_dir}/selected_features.cpkl", "wb") as features_file:
        cloudpickle.dump([features[fid] for fid in fids], features_file)
        cloudpickle.dump(windows, features_file)

    logger.info("Loading data")
    train_set = np.load(f"{args.data_dir}/{SAVED_DIR}/{ext_utils.TRAIN}_{ext_utils.DATA_FILENAME}")
    train_data = prune_data(train_set["arr_0"], fids, windows)
    test_set = np.load(f"{args.data_dir}/{SAVED_DIR}/{ext_utils.TEST}_{ext_utils.DATA_FILENAME}")
    test_data = prune_data(test_set["arr_0"], fids, windows)
    val_set = np.load(f"{args.data_dir}/{SAVED_DIR}/{ext_utils.VAL}_{ext_utils.DATA_FILENAME}")
    val_data = prune_data(val_set["arr_0"], fids, windows)

    logger.info("Writing data")
    np.savez(f"{args.output_dir}/{ext_utils.TRAIN}_{ext_utils.DATA_FILENAME}", train_data, train_set["arr_1"])
    np.savez(f"{args.output_dir}/{ext_utils.TEST}_{ext_utils.DATA_FILENAME}", test_data, test_set["arr_1"])
    np.savez(f"{args.output_dir}/{ext_utils.VAL}_{ext_utils.DATA_FILENAME}", val_data, val_set["arr_1"])


def select_features(args, features):
    """Select features to use"""
    num_features = len(features)
    fids = [idx for idx in range(num_features) if features[idx].important]
    num_new_features = len(fids)
    windows = [None] * num_new_features
    for idx, fid in enumerate(fids):
        window = features[fid].temporal_window
        assert window is not None
        windows[idx] = window
    if args.randomize_features:
        fids = np.random.default_rng(args.seed).choice(range(num_features), size=num_new_features, replace=False)
    return fids, windows


def prune_data(data, fids, windows):
    """Prune data according to feature importance analysis"""
    # data: instances X timesteps X features
    assert data.shape[2] > max(fids)
    new_data = data[:, :, fids]
    for idx in range(len(fids)):
        left, right = windows[idx]
        new_data[:, 0: left, idx] = 0
        new_data[:, right + 1:, idx] = 0
    return new_data


if __name__ == "__main__":
    main()
