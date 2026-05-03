#!/bin/bash
#SBATCH --job-name=cub_slot_finetune
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-cub_slot_finetune.out
#SBATCH --error=slurm-%j-cub_slot_finetune.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

python finetune_slots.py \
    --n_epochs 300 \
    --lr 2e-4 \
    --batch_size 128 \
    --sa_iters 3 \
    --entropy_weight 0.0 \
    --tv_weight 0.0 \
    --viz_every 5 \
    --n_slots 50 \
    --dinosaur_ckpt "checkpoints/dinov3/epoch_67-step_500000_dinov3_50_slots.ckpt" \
    --out_dir "cub_dinosaurv3_finetune_50_slots" 
