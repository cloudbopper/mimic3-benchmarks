"""Utility functions for extending mimic-3 benchmark functionality"""


import os
import pickle


import numpy as np

# File names
SAVED = "saved"  # Directory where outputs are saved
MODEL_FILENAME = "model.h5"  # HDF5 file: see https://www.tensorflow.org/tutorials/keras/save_and_load
DATA_FILENAME = "data.npz"  # numpy arrays
DISCRETIZER_FILENAME = "discretizer.pkl"
DISCRETIZER_HEADER_FILENAME = "discretizer_header.pkl"
CONT_CHANNELS_FILENAME = "cont_channels.pkl"
NAMES_FILENAME = "names.pkl"

# Data types
TRAIN = "train"
VAL = "val"
TEST = "test"


def initialize_saved(output_dir):
    """Initialize saved directory"""
    path = os.path.join(output_dir, SAVED)
    if not os.path.exists(path):
        os.makedirs(path)
    


def write_model(model, output_dir):
    """Write model to file"""
    model_path = os.path.join(output_dir, SAVED, MODEL_FILENAME)
    model.save(model_path)


def write_data(data, output_dir, dtype=TRAIN):
    """Write data to file"""
    assert dtype in {TRAIN, VAL, TEST}, f"Invalid data type {dtype}"
    data_path = os.path.join(output_dir, SAVED, f"{dtype}_{DATA_FILENAME}")
    features, labels = data  # Feature matrix/tensor, labels
    np.savez(data_path, features, labels)


def write_optional(obj, output_dir, filename):
    """Write optional file"""
    path = os.path.join(output_dir, SAVED, filename)
    with open(path, "wb") as filep:
        pickle.dump(obj, filep)
