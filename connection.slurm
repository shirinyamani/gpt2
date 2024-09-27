#!/bin/bash
echo "Hi there! Trial for allocation 1 GPU A100 for 10h started..."

#! /bin/bash
# ====================================
#SBATCH --job-name=gpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
# ====================================


####### Set environment variables ###############
module load python/3.10.4
module load cuda/12.1.1

####### Run your script #########################
python bigram.py 