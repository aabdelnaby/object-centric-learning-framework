#!/bin/bash
#SBATCH --job-name=movi_c_feat_rec_dino_a100_long_4gpus_hierarchical_slots
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24          
#SBATCH --gres=gpu:4           
#SBATCH --time=48:00:00          
#SBATCH --output=slurm_jobs_logs/train_short_movi_hierarchical_slots/slurm-%j-movi_c_feat_rec_dino.out
#SBATCH --error=slurm_jobs_logs/train_short_movi_hierarchical_slots/slurm-%j-movi_c_feat_rec_dino.err

source ~/.bashrc
conda activate oclf_env   

export DATASET_PREFIX=scripts/datasets/outputs
poetry run ocl_train \
    +experiment=projects/bridging/dinosaur/movi_c_feat_rec \
    hydra.job.chdir=false 
    
