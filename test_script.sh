#!/usr/bin/env bash
#SBATCH --job-name=test-job
#SBATCH --nodes=1
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --verbose

echo "hello"
