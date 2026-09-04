#!/bin/bash
# Train the hier_router on the COMBINED (ADE20K + COCO/PACO) dataset to convergence.
#
# PACO is now in train+val, so the model actually learns COCO parts (not just transfer).
# Uses the precomputed combined caches (frozen DINOv3 + T5-spans) so each epoch is fast.
# gpu_a100_il (long partition) to avoid the 30-min cap; best_model.pt is saved on every
# val improvement and a checkpoint every --checkpoint_every epochs (so a timeout/--resume
# loses nothing). >200 unique queries → per-query eval auto-skips (also forced below).
#
#   sbatch submit_train_combined.sh
# Quick partial signal instead: switch --partition=gpu_a100_short --time=00:30:00.

#SBATCH --job-name=comb_hr
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j-combined_train.out
#SBATCH --error=slurm-%j-combined_train.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

python train.py \
    --dataset        ade20k \
    --csv_path       FG-datset/parts_color_vqa_combined.csv \
    --pooler         hier_router \
    --text_encoder   t5 \
    --feat_cache \
    --dino_cache     FG-datset/dino_feat_cache_combined_square.pt \
    --text_cache     FG-datset/text_feat_cache_combined_t5_spans.pt \
    --dinosaur_cfg   projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3 \
    --dinosaur_ckpt  checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt \
    --n_slots        11 \
    --resize_mode    square \
    --optimizer      adam \
    --lr             1e-4 \
    --lr_schedule    constant \
    --batch_size     128 \
    --max_epochs     300 \
    --checkpoint_every 25 \
    --checkpoint_dir runs/combined_hier_router \
    --skip_per_query_eval
