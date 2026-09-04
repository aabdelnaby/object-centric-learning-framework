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
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j-scvqa_validate_object_color.out
#SBATCH --error=slurm-%j-scvqa_validate_object_color.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# T-n downstream depth; TF-2 is fast and sufficient for a sanity check.
POOLER_LAYERS=5

CSV=FG-datset/superclevr3d/depth1_color_synthesized.csv
DINO_CKPT=checkpoints/dinov3/vitb_448_epoch_320-step_100000.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_base16_dinov3_448
# Full ViT-B/448 n5 cache (already on disk; covers all sanity images). Using it
# directly avoids a redundant ~10GB subcache copy (we're tight on /pfs quota).
DINO_CACHE=FG-datset/superclevr3d/dino_feat_cache_vitb_448_simple_color_d2_n5.pt
CKPT_DIR=runs/superclevr_vqa_sanity_T${POOLER_LAYERS}_object_color/long_fr/12slots

# ── Train + validate (per-epoch val accuracy is the sanity signal) ───────────
# Text features are computed in RAM at startup (--text_in_memory); the ViT-B/448
# image features come from the existing on-disk dino cache (--feat_cache).
python train.py \
    --dataset          superclevr3d \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           vqa_paper \
    --pooler_layers    "$POOLER_LAYERS" \
    --n_slots          12 \
    --dinosaur_cfg     "$DINO_CFG" \
    --dinosaur_ckpt    "$DINO_CKPT" \
    --img_size         448 \
    --feat_cache \
    --text_in_memory \
    --dino_in_memory \
    --batch_size       128 \
    --optimizer        adam \
    --lr_schedule      constant \
    --lr               5e-5 \
    --warmup_steps     10000 \
    --max_steps        60000 \
    --patience         1000000 \
    --checkpoint_every 20 \
    --viz_n_samples 4 \
    --num_heads        64 \
    --skip_per_query_eval \
    --checkpoint_dir   "$CKPT_DIR" 
