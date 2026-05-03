#!/bin/bash
# Submit a single n_slots training run as a Slurm job.
#
# Step 1 — precompute ViT + RoBERTa features (once, ~20 min on A100):
#   sbatch submit_precompute.sh
#
# Step 2 — launch the run (after the precompute job finishes):
#   sbatch submit_slot_single.sh
#
# Results land in:
#   cub_classifier_checkpoints/slots_<N>/best_model.pt
#   cub_classifier_checkpoints/slots_<N>/metrics.csv

#SBATCH --job-name=cub_slots_single
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slots_exps/color_all/50_slots_long_24h/slurm-%j-slots.out
#SBATCH --error=slots_exps/color_all/50_slots_long_24h/slurm-%j-slots.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

N_SLOTS=50

echo "=== Single job  |  n_slots=$N_SLOTS ==="

python train.py \
    --n_slots      $N_SLOTS \
    --max_epochs   5000 \
    --batch_size   256 \
    --warmup_steps 200 \
    --patience      5000 \
    --category_filter "color" \
    --feat_cache \
    --checkpoint_every 20 \
    --dinosaur_ckpt "/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/checkpoints/dinov3/epoch_67-step_500000_dinov3_50_slots.ckpt" \
    --checkpoint_dir "cub_classifier_checkpoints_color_all/50_slots_long_ft_24hr/slots_${N_SLOTS}" \
    --finetune_ckpt "/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/cub_dinosaurv3_finetune_50_slots/best_checkpoint.pt"