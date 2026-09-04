#!/bin/bash
# Precompute the CUB DINOv3 image-feature cache ONCE for the hier_router run.
#
#   FG-datset/CUB_200_2011/dino_feat_cache_dinov3_square.pt
#     frozen DINOv3 ViT-S/16 features at 224, SQUARE resize (Resize((224,224)),
#     no center crop) — the same backbone + preprocessing the ADE20K hier_router
#     was tuned on. 200 tokens/image (4 register + 196 patch), fp16.
#     These are n_slots-INDEPENDENT (the ViT runs before slot attention), so the
#     trainer re-runs slot attention live on them for whatever --n_slots it uses.
#
# Text is NOT cached here: CUB has only 28 unique queries, so the trainer builds
# the T5 hidden states + the 4 hier_router span channels (with the CUB parser)
# in RAM at startup via --text_in_memory (instant). Run this FIRST, then
# sbatch submit_cub_hier_router.sh.
#
#   sbatch submit_precompute_cub_dinov3.sh

#SBATCH --job-name=cubprecomp
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-cub_precompute_dinov3.out
#SBATCH --error=slurm-%j-cub_precompute_dinov3.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

CSV=FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv
IMG_ROOT=FG-datset/CUB_200_2011/images
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt

python precompute_features.py \
    --csv            "$CSV" \
    --image_root     "$IMG_ROOT" \
    --dino_cfg       "$DINO_CFG" \
    --dino_ckpt      "$DINO_CKPT" \
    --n_slots        7 \
    --img_size       224 \
    --resize_mode    square \
    --batch_size     64 \
    --num_workers    12 \
    --fp16 \
    --skip_text \
    --dino_cache_out FG-datset/CUB_200_2011/dino_feat_cache_dinov3_square.pt
