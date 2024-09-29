#! /bin/bash
# ====================================
#SBATCH --job-name=gpt-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
# ====================================

python bigram.py