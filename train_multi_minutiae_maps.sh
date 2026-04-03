#!/usr/bin/env bash
#SBATCH --job-name=multi_minutiae_fingerprint_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --verbose


source /home/khangphuanhle/miniforge3/etc/profile.d/conda.sh
conda activate fml
python /home/khangphuanhle/fingerprint-generator/fingerprint_net_v2/exp_train_multi_minutiae.py

