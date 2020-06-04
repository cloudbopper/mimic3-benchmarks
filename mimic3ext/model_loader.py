"""Load/save Tensorflow model"""

import argparse
import os
import imp
import sys

import numpy as np

from mimic3models import common_utils
from mimic3ext import ext_utils, ModelWrapper


MODULE_DIR = f"{os.path.dirname(__file__)}/.."
DEFAULT_NETWORK_FILE = f"{MODULE_DIR}/mimic3models/keras_models/lstm.py"
def build_model(state_filename="",
                strargs=(f"--network {DEFAULT_NETWORK_FILE} "
                         "--dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8"),
                input_dim=None):
    """
    Build in-hospital-mortality LSTM model
    Tried to make it as identical as possible to mimic3models/in_hospital_mortality/main.py
    """
    # Args..
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    args = parser.parse_args(strargs.split(" "))

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

    # More args..
    args_dict = dict(args._get_kwargs())
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl

    # Input dimension
    if input_dim is not None:
        args_dict["input_dim"] = input_dim

    # Build the model
    print("==> using model {}".format(args.network))
    model_module = imp.load_source(os.path.basename(args.network), args.network)
    model = model_module.Network(**args_dict)
    suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                    ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                    ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                    args.timestep,
                                    ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
    model.final_name = args.prefix + model.say_name() + suffix
    print("==> model.final_name:", model.final_name)


    # Compile the model
    print("==> compiling the model")
    optimizer_config = {'class_name': args.optimizer,
                        'config': {'lr': args.lr,
                                'beta_1': args.beta_1}}

    # NOTE: one can use binary_crossentropy even for (B, T, C) shape.
    #       It will calculate binary_crossentropies for each class
    #       and then take the mean over axis=-1. Tre results is (B, T).
    if target_repl:
        loss = ['binary_crossentropy'] * 2
        loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
    else:
        loss = 'binary_crossentropy'
        loss_weights = None

    model.compile(optimizer=optimizer_config,
                loss=loss,
                loss_weights=loss_weights)
    model.summary()

    if state_filename:
        # Load weights
        model.load_weights(state_filename)

    return model

def load_model(model_filename):
    """Load model from file"""
    tf_model = build_model(state_filename=model_filename)
    return ModelWrapper(tf_model)


def save_model(model, model_filename):
    """Save model to file"""
    model.model.save_weights(model_filename)
