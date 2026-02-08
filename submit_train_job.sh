#!/bin/bash
#SBATCH --job-name=train_coco_hierarchical_dinov3
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24          
#SBATCH --gres=gpu:1           
#SBATCH --time=48:00:00
#SBATCH --output=slurm_jobs_logs/train/dinov3/long/slurm-%j.out
#SBATCH --error=slurm_jobs_logs/train/dinov3/long/slurm-%j.err

source ~/.bashrc
conda activate oclf_env   

export DATASET_PREFIX=scripts/datasets/outputs
poetry run ocl_train \
    +experiment=projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto_dinov3 \
    hydra.job.chdir=false 
    