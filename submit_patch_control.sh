#!/bin/bash
# Submit the PatchClassifier control experiment (raw ViT patches, no slot attention).
#
# Requires precomputed features — run precompute_features.py first if needed:
#   sbatch submit_precompute.sh
#
# Results land in:
#   cub_classifier_checkpoints_patch_control/patches/best_model.pt
#   cub_classifier_checkpoints_patch_control/patches/metrics.csv

#SBATCH --job-name=cub_patch_ctrl
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slots_exps/patch_control/slurm-%A.out
#SBATCH --error=slots_exps/patch_control/slurm-%A.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

mkdir -p slots_exps/patch_control

echo "=== PatchClassifier control experiment ==="

python train.py \
    --patch_control \
    --max_epochs   300 \
    --batch_size   64 \
    --warmup_steps 200 \
    --patience     300 \
    --category_filter "color" \
    --feat_cache \
    --checkpoint_every 5 \
    --dinosaur_ckpt "/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/checkpoints/dinov3/epoch_20-step_155206.ckpt" \
    --checkpoint_dir "cub_classifier_checkpoints_patch_control"
