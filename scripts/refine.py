import csv
import pickle
import numpy as np


# Run from project root directory as "python -m scripts.refine"
DATA_FILENAME = "data/preprocessed_lstm/in-hospital-mortality.pkl"
QUANTIZATION_LEVELS = "4"  # for continuous variables 


with open(DATA_FILENAME, "rb") as data_file:
    train_raw = pickle.load(data_file)
    discretizer_header = pickle.load(data_file)
    cont_channels = pickle.load(data_file)
    discretizer = pickle.load(data_file)

    # Write header
    feature_names = list(discretizer._channel_to_id.keys())
    feature_types = list(map(lambda name: "C" if not discretizer._possible_values[name] else "D", feature_names))
    feature_numvalues = list(map(lambda name: QUANTIZATION_LEVELS if not discretizer._possible_values[name]
                                 else str(len(discretizer._possible_values[name])), feature_names))
    with open("in-hospital-mortality.header", "w") as header_file:
        writer = csv.writer(header_file, delimiter=" ")
        writer.writerow([name.replace(" ", "-") for name in feature_names])
        writer.writerow(feature_numvalues)
        writer.writerow(feature_types)

    # Write data
    discrete = np.array(feature_types) == "D"
    num_features = len(discrete)
    data_file = open("in-hospital-mortality.data", "w")
    writer = csv.writer(data_file, delimiter=" ")
    data = train_raw[0]
    num_sequences = data.shape[0]
    for sidx, sequence in enumerate(data):
        if sidx % 100 == 0:
            print("Processing sequence index %d of %d" % (sidx + 1, num_sequences))
        for feature_vector in sequence:
            cidx = 0
            row = np.empty(num_features, dtype="<U8")
            for fidx, name in enumerate(feature_names):
                is_discrete = discrete[fidx]
                numvals = int(feature_numvalues[fidx]) if is_discrete else 1
                if is_discrete:
                    row[fidx] = np.argwhere(feature_vector[cidx: cidx + numvals])[0][0]
                else:
                    row[fidx] = np.around(feature_vector[cidx], decimals=4)
                cidx += numvals
            writer.writerow(row)
