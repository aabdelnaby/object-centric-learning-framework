#!/bin/bash
#SBATCH --job-name=slurm-%j-eval
#SBATCH --partition=gpu_a100_short       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24          
#SBATCH --gres=gpu:4           
#SBATCH --time=30             
#SBATCH --output=slurm_jobs_logs/eval/hierarchy/139867steps_2slots_short_sa_gating/slurm-%j-movi_c_feat_rec_dino_a100short4gpu.out
#SBATCH --error=slurm_jobs_logs/eval/hierarchy/139867steps_2slots_short_sa_gating/slurm-%j-movi_c_feat_rec_dino_a100short4gpu.err

source ~/.bashrc
conda activate oclf_env   

export DATASET_PREFIX=scripts/datasets/outputs
poetry run ocl_eval \
  +train_config_path=/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/outputs/projects/bridging/dinosaur/movi_c_feat_rec/2025-09-15_10-54-00/config/config.yaml \
  +evaluation=projects/bridging/outputs_movi_e.yaml \
  +checkpoint_path=/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/checkpoints/epoch_0-step_7361_hierarchy_2_slots_sa_gating.ckpt \
  +output_dir=.
