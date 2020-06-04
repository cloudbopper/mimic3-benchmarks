"""Learn in-hospital-mortality model from tensor data"""
import argparse
import os

from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np

from mimic3ext import ext_utils
from mimic3ext.model_loader import build_model
from mimic3models import keras_utils
from mimic3models import metrics
from mimic3models.in_hospital_mortality import utils

BATCH_SIZE = 8
SAVE_EVERY = 1
VERBOSE = 2
EPOCHS = 100


def main():
    """Train/test model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", required=True)
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-state_filename", help="If provided, load model weights from state file instead of training model")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load or train model
    if args.state_filename:
        model = build_model(args.state_filename)
    else:
        model = build_model()
        train(args, model)
    # Test model
    test(args, model)
    

def train(args, model):
    """Train model"""
    train_filename = f"{args.input_dir}/{ext_utils.TRAIN}_{ext_utils.DATA_FILENAME}"
    train_raw = list(np.load(train_filename).values())
    val_filename = f"{args.input_dir}/{ext_utils.VAL}_{ext_utils.DATA_FILENAME}"
    val_raw = list(np.load(val_filename).values())

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=False,
                                                              batch_size=BATCH_SIZE,
                                                              verbose=VERBOSE)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=SAVE_EVERY)

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=EPOCHS,
              initial_epoch=0,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=2,
              batch_size=8)

    return model


def test(args, model):
    """Test model"""
    test_filename = f"{args.input_dir}/{ext_utils.TEST}_{ext_utils.DATA_FILENAME}"
    data, labels = list(np.load(test_filename).values())
    names = [f"Feature {idx}" for idx in range(data.shape[2])]
    

    predictions = model.predict(data, batch_size=BATCH_SIZE, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test_predictions.csv")
    utils.save_results(names, predictions, labels, path)


if __name__ == "__main__":
    main()
