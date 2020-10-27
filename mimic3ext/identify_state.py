"""Identify model state with lowest validation set loss"""
import argparse
import os
import re

import numpy as np

from mimic3ext import ext_utils

LAST = "last"
BEST = "best"

def main():
    """Identify model state with lowest validation set loss"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-states_dir", required=True)
    parser.add_argument("-mode", choices=[LAST, BEST], required=True)
    parser.add_argument("-epochs", type=int, default=ext_utils.EPOCHS)
    parser.add_argument("-debug", action="store_true")
    args = parser.parse_args()
    return pipeline(args)


def pipeline(args):
    """Pipeline"""
    state_filenames = os.listdir(args.states_dir)
    epochs = np.zeros(args.epochs)
    losses = np.zeros(args.epochs)
    for idx, state_filename in enumerate(state_filenames):
        match = re.search(r"epoch(\d+)\.test(\d+\.\d+)\.state$", state_filename)
        epoch, loss = match.groups()
        epochs[idx] = int(epoch)
        losses[idx] = float(loss)
    last_idx = np.argmax(epochs)
    last_epoch = epochs[last_idx]
    last_state = state_filenames[last_idx]
    if args.debug:
        print(f"Last state: {last_state} for epoch {last_epoch}")
    best_idx = np.argmin(losses)
    best_epoch = epochs[best_idx]
    best_state = state_filenames[best_idx]
    if args.debug:
        print(f"Best state: {best_state} for epoch {best_epoch}")
    if args.mode == LAST:
        print(last_state)
        return last_state, last_epoch
    elif args.mode == BEST:
        print(best_state)
        return best_state, best_epoch
    else:
        raise ValueError("Undefined mode")


if __name__ == "__main__":
    main()
