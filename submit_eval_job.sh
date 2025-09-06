#!/bin/bash
#SBATCH --job-name=slurm-%j-eval
#SBATCH --partition=gpu_a100_short       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24          
#SBATCH --gres=gpu:4           
#SBATCH --time=30             
#SBATCH --output=slurm_jobs_logs/slurm-%j-movi_c_feat_rec_dino_a100short4gpu.out
#SBATCH --error=slurm_jobs_logs/slurm-%j-movi_c_feat_rec_dino_a100short4gpu.err

source ~/.bashrc
conda activate oclf_env   

export DATASET_PREFIX=scripts/datasets/outputs
poetry run ocl_eval \
  +train_config_path=/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/outputs/projects/bridging/dinosaur/movi_c_feat_rec/2025-09-05_15-11-10/config/config.yaml \
  +evaluation=projects/bridging/outputs_movi_e.yaml \
  +checkpoint_path=/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/epoch_98-step_90924.ckpt \
  +output_dir=.
