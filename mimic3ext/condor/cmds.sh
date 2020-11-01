#!/bin/bash

mkdir wdir
mv cmd_learn.sh pruned.tar.gz wdir
cd wdir

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
bash ./miniconda.sh -b -p miniconda
eval "$(./miniconda/bin/conda shell.bash hook)"
conda create -y --name mimic
conda activate mimic
conda install -y numpy pandas scikit-learn keras-gpu=2.3.1 tensorflow-gpu

git clone https://github.com/cloudbopper/mimic3-benchmarks.git
cd mimic3-benchmarks
mkdir data
tar -xvzf ../pruned.tar.gz -C data/

python -um mimic3ext.learn -input_dir data -output_dir ../../outputs -epochs 10 |& tee ../../train_test.log
best_state=$(python -um mimic3ext.identify_state -states_dir ../../outputs/keras_states/ -mode best -epochs 10)
python -um mimic3ext.learn -input_dir data -output_dir ../../outputs -state_filename ../../outputs/keras_states/${best_state} |& tee ../../test_best.log

cd ../..
tar -cvzf outputs.tar.gz outputs
