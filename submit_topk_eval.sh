#!/bin/bash
#SBATCH --job-name=topkEval
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-topk_eval.out
#SBATCH --error=slurm-%j-topk_eval.err

source ~/.bashrc
conda activate oclf_env
cd /home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework
python -u _topk_eval.py
