#!/usr/bin/env bash
#SBATCH --job-name=fprint_training_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --verbose


source /home/khangphuanhle/.bashrc
conda activate fml
python /home/khangphuanhle/fingerprint-generator/fingerprint_net_v2/exp_train_with_freq.py

