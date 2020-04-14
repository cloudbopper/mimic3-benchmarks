#!/bin/bash

mkdir wdir
mv cmds.sh in-hospital-mortality.tar.gz wdir
cd wdir

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
bash ./miniconda.sh -b -p miniconda
eval "$(./miniconda/bin/conda shell.bash hook)"
conda create -y --name mimic
conda activate mimic
conda install -y numpy pandas scikit-learn keras-gpu tensorflow-gpu

git clone https://github.com/cloudbopper/mimic3-benchmarks.git
cd mimic3-benchmarks
mkdir data
tar -xvzf ../in-hospital-mortality.tar.gz -C data/

python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality --verbose 1 --small_part |& tee ../../train.log

python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models/in_hospital_mortality/keras_states/*epoch100.* --verbose 1 --small_part |& tee ../../test.log

cd ../..
tar -cvzf saved.tar.gz wdir/mimic3-benchmarks/mimic3models/in_hospital_mortality/saved
tar -cvzf saved_model_states.tar.gz wdir/mimic3-benchmarks/mimic3models/in_hospital_mortality/keras_states
