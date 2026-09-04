#!/bin/bash
# Hierarchical slot router (pooler=hier_router) on CUB-200.
#
# Tests the ADE20K parent->child path-marginalisation head on CUB's
# "What is the <part> <attribute> of the bird?" questions, with:
#   - object <y> = "bird"           -> parent routing  P(j|y)
#   - part   <x> = the body part    -> child  routing  P(k|j,x)
#   - readout = "bird <part> <attribute>" (e.g. "bird back color")
#         -> child routing + the answer readout. The attribute word lives ONLY on
#            this channel, so the head can tell "back color" from "back pattern"
#            (both route to the same part). See precompute_features.parse_cub_xy_phrases.
#
# Same backbone + recipe as the proven ADE20K readout run
# (submit_ade20k_hier_router_readout.sh): DINOv3 ViT-S/16, square resize. The
# answer head is generic (log P(a) over the CUB label vocab) — no head change.
#
# Prereq: sbatch submit_precompute_cub_dinov3.sh  (builds the DINO image cache).
# Text (T5 + the 4 CUB span channels) is built in RAM at startup (--text_in_memory;
# 28 queries -> instant). Per-attribute val accuracy prints (28 queries <= 200).
#
#   sbatch submit_cub_hier_router.sh

#SBATCH --job-name=cubRDO
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j-cub_hier_router.out
#SBATCH --error=slurm-%j-cub_hier_router.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

POOLER_LAYERS=5

CSV=FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
DINO_CACHE=FG-datset/CUB_200_2011/dino_feat_cache_dinov3_square.pt
# Per-job checkpoint dir (SLURM_JOB_ID; falls back to "local" outside SLURM).
CKPT_DIR=runs/cub_hier_router/${SLURM_JOB_ID:-local}

python train.py \
    --dataset          cub \
    --category_filter "color" \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --rank_method      attribution \
    --pooler_layers    "$POOLER_LAYERS" \
    --n_slots          7 \
    --dinosaur_cfg     "$DINO_CFG" \
    --dinosaur_ckpt    "$DINO_CKPT" \
    --img_size         224 \
    --resize_mode      square \
    --feat_cache \
    --dino_cache       "$DINO_CACHE" \
    --text_in_memory \
    --batch_size       128 \
    --lr               2e-4 \
    --warmup_steps     2000 \
    --max_steps        500000 \
    --patience         1000000 \
    --checkpoint_every 30 \
    --viz_n_samples    20 \
    --num_heads        64 \
    --recursive_children 5 \
    --recursive_spread 0.0 \
    --pooler_dropout   0.0 \
    --router_temp      0.95 \
    --child_scorer     mlp \
    --weight_decay     0.02 \
    --checkpoint_dir   "$CKPT_DIR"
