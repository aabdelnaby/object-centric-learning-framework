#!/bin/bash
#SBATCH --job-name=cub_classifier_test
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-cub_classifier_test.out
#SBATCH --error=slurm-%j-cub_classifier_test.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

python train.py \
    --query_filter "What is the wing color of the bird?" \
    --epochs 3 \
    --batch_size 32 \
    --warmup_steps 100
