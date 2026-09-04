#!/bin/bash
# Precompute the ADE20K feature caches ONCE for the hier_router Optuna study.
#
#   - dino_feat_cache_square.pt : frozen DINOv3 ViT features at 224, SQUARE resize
#     (Resize((224,224)), no center crop). These are n_slots-INDEPENDENT (the ViT
#     runs before slot attention), so every Optuna trial — whatever n_slots/K it
#     samples — reuses this one cache. Distinct path from the crop cache.
#   - text_feat_cache_t5_spans.pt : T5-base hidden states + x_vec/y_vec span vectors
#     (--with_spans) for the "color of <x> of <y>" questions, required by hier_router.
#
# Run this FIRST, then sbatch submit_tune_hier_router.sh.
#
#   sbatch submit_precompute_ade20k_square.sh

#SBATCH --job-name=adeprecomp
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ade20k_precompute_square.out
#SBATCH --error=slurm-%j-ade20k_precompute_square.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

CSV=FG-datset/ade20k/parts_color_vqa_internvl.csv
IMG_ROOT=FG-datset/ade20k/ADE20K_2021_17_01/images
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt

python precompute_features.py \
    --csv            "$CSV" \
    --image_root     "$IMG_ROOT" \
    --dino_cfg       "$DINO_CFG" \
    --dino_ckpt      "$DINO_CKPT" \
    --n_slots        11 \
    --img_size       224 \
    --resize_mode    square \
    --text_encoder   t5 \
    --with_spans \
    --dino_cache_out FG-datset/ade20k/dino_feat_cache_square.pt \
    --text_cache_out FG-datset/ade20k/text_feat_cache_t5_spans.pt
