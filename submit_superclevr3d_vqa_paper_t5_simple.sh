#!/bin/bash
# SANITY VALIDATION of the paper-faithful VQA classifier on the STANDARD
# Super-CLEVR VQA questions (CLEVR-style: count/exist/compare/query-attribute),
# NOT the Super-CLEVR-3D part questions.
#
# Goal: confirm the model actually learns Super-CLEVR VQA before committing to a
# full run — val accuracy should climb well above the majority-class baseline.
#
# Data (subset for a quick check):
#   FG-datset/superclevr3d/superclevr_vqa_sanity.csv
#     5,000 images (n_objects<=5), 50,000 questions (40k train / 10k val),
#     ~49k unique questions, 41 train answer classes.
#     val majority-class ("False") baseline = 21.6% ; chance = 2.4%.
#   Built from Super-CLEVR/superCLEVR_questions_30k.json, restricted to the
#   images covered by the ViT-B/448 DINOv3 feature cache.
#
# Model = the same paper-faithful classifier as the main experiment:
#   upstream  : frozen DINOSAUR slots, DINOv3 ViT-B/16 @448, 24 slots,
#               ckpt checkpoints/dinov3/vitb-448-epoch_272-step_85000.ckpt
#   text      : T5-base encoder (768-d), frozen
#   downstream: --pooler vqa_paper (d_model=128, ff=128, 64 heads, MLP head)
#   training  : Adam, constant lr 1e-4, batch 128, cross-entropy (paper App. A.3)
#
# SUCCESS CRITERION: val acc >> 21.6% (majority baseline). If it stays near
# ~21-25%, the model isn't using the image (wiring/feature problem).
#
# Text features: --text_in_memory encodes the ~49k unique questions with T5 once
# at run start and keeps them in RAM (shared by train+val). Nothing is written to
# disk, so no ~10GB text cache and no /pfs quota / corrupt-partial-write risk.
#
# Submit:
#   sbatch submit_superclevr_validate_t5_vqa_paper.sh
# Resumable via --resume.

#SBATCH --job-name=scvqa_validate
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-scvqa_validate_cub_part_color_recursive.out
#SBATCH --error=slurm-%j-scvqa_validate_cub_part_color_recursive.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# T-n downstream depth; TF-2 is fast and sufficient for a sanity check.
POOLER_LAYERS=2

CSV=FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv
# DINO_CKPT=checkpoints/dinov3/vitb_24slots_epoch=368-step=115000.ckpt
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_50_slots.ckpt
# DINOv3 ViT-S/16 @224 (384-dim) — matches the epoch_21 checkpoint above.
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# NOTE: the on-disk cache below is ViT-B/448 (768-dim) and is INCOMPATIBLE with the
# ViT-S/16 @224 backbone above. It is unused here: --dino_in_memory recomputes the
# S/16@224 features in RAM and bypasses the disk cache. Left for reference only.
# Per-job checkpoint dir (SLURM_JOB_ID; falls back to "local" outside SLURM).
CKPT_DIR=runs/recursive_cub_T${POOLER_LAYERS}_part_color/topk/${SLURM_JOB_ID:-local}

# ── Train + validate (per-epoch val accuracy is the sanity signal) ───────────
# Text features (T5) and DINOv3 ViT-S/16 @224 image features are both computed in
# RAM at startup (--text_in_memory / --dino_in_memory); nothing is read from disk.
python train.py \
    --dataset          cub \
    --category_filter "color" \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           vqa_paper \
    --pooler_layers    "$POOLER_LAYERS" \
    --n_slots          5 \
    --dinosaur_cfg     "$DINO_CFG" \
    --dinosaur_ckpt    "$DINO_CKPT" \
    --img_size         224 \
    --feat_cache \
    --text_in_memory \
    --dino_in_memory \
    --batch_size       224 \
    --lr               1e-4 \
    --warmup_steps     10000 \
    --max_steps        60000 \
    --patience         1000000 \
    --checkpoint_every 5 \
    --viz_n_samples 4 \
    --num_heads        64 \
    --skip_per_query_eval \
    --recursive_infer \
    --recursive_children 3 \
    --recursive_parents 1 \
    --recursive_spread 0.0 \
    --train_frac 0.5 \
    --pooler_dropout 0.2 \
    --checkpoint_dir   "$CKPT_DIR" 
