#!/bin/bash
# Smoke-test the hier_router on the COMBINED (ADE20K + COCO/PACO) CSV.
#
# Purpose: confirm the combined CSV trains end-to-end — PACO rows load (absolute
# image_name), PACO queries parse into <x>/<y> spans, the split column drives
# train/val, and the 12-class label vocab covers both domains. Short run
# (--max_epochs 2, no checkpoints) off the precomputed combined cache.
#
# Needs: submit_precompute_combined_square.sh to have produced the caches first.
#   sbatch submit_train_combined_smoke.sh

#SBATCH --job-name=combsmoke
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-combined_smoke.out
#SBATCH --error=slurm-%j-combined_smoke.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

python train.py \
    --dataset        ade20k \
    --csv_path       FG-datset/parts_color_vqa_combined.csv \
    --pooler         hier_router \
    --feat_cache \
    --dino_cache     FG-datset/dino_feat_cache_combined_square.pt \
    --text_cache     FG-datset/text_feat_cache_combined_t5_spans.pt \
    --dinosaur_cfg   projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3 \
    --dinosaur_ckpt  checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt \
    --n_slots        11 \
    --text_encoder   t5 \
    --resize_mode    square \
    --optimizer      adam \
    --lr             1e-4 \
    --lr_schedule    constant \
    --batch_size     128 \
    --max_epochs     2 \
    --checkpoint_every 0
