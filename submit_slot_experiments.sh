#!/bin/bash
# Submit the n_slots sweep as a Slurm job array.
#
# Step 1 — precompute ViT + RoBERTa features (once, ~20 min on A100):
#   sbatch submit_precompute.sh
#
# Step 2 — launch the sweep (after the precompute job finishes):
#   sbatch submit_slot_experiments.sh
#
# Results land in:
#   cub_classifier_checkpoints/slots_<N>/best_model.pt
#   cub_classifier_checkpoints/slots_<N>/metrics.csv

#SBATCH --job-name=cub_slots
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=slots_exps/color_all/100_slots_frfr/slurm-%A-%a-slots.out
#SBATCH --error=slots_exps/color_all/100_slots_frfr/slurm-%A-%a-slots.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# Slot counts to sweep — one job-array task per entry
SLOT_COUNTS=(35 50 100)
N_SLOTS=${SLOT_COUNTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Job array task $SLURM_ARRAY_TASK_ID  |  n_slots=$N_SLOTS ==="

python train.py \
    --n_slots      $N_SLOTS \
    --max_epochs   5000 \
    --batch_size   64 \
    --warmup_steps 200 \
    --patience      5000 \
    --category_filter "color" \
    --feat_cache \
    --checkpoint_every 5 \
    --dinosaur_ckpt "/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/checkpoints/dinov3/epoch_67-step_500000_dinov3_100_slots.ckpt" \
    --checkpoint_dir "cub_classifier_checkpoints_color_all/100_slots_attention_weighted_pooling/slots_${N_SLOTS}"
