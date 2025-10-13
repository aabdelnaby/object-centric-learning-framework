#!/bin/bash
#SBATCH --job-name=train_coco_hierarchical
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24          
#SBATCH --gres=gpu:4           
#SBATCH --time=48:00:00
#SBATCH --output=slurm_jobs_logs/train/movi_hierarchical_coco_long/slurm-%j-movi_c_feat_rec_dino_dev.out
#SBATCH --error=slurm_jobs_logs/train/movi_hierarchical_coco_long/slurm-%j-movi_c_feat_rec_dino_dev.err

source ~/.bashrc
conda activate oclf_env   

export DATASET_PREFIX=scripts/datasets/outputs
poetry run ocl_train \
    +experiment=projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto \
    hydra.job.chdir=false 
    