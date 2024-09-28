#!/bin/bash
echo "Hi there! Trial for allocation 1 GPU A100 for 10h started..."

#! /bin/bash
# ====================================
#SBATCH --job-name=bigram
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
# ====================================

# Check the GPUs with the nvidia-smi command.
nvidia-smi
# ====================================
# Report allocated nodes.
echo "Alocated nodes: $SLURM_JOB_NODELIST"

####### Set environment variables ###############
module load python/3.10.4
module load cuda/12.1.1

####### Run your script #########################
python bigram.py 