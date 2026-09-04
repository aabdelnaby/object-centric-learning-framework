#!/bin/bash
# Precompute the COMBINED (ADE20K + COCO/PACO) feature caches for the hier_router.
#
# Same DINOv3 (square-224) + T5-span recipe as submit_precompute_ade20k_square.sh,
# just over the combined CSV. PACO rows carry ABSOLUTE image_name, so they resolve
# regardless of --image_root (os.path.join ignores the root for absolute paths);
# ADE rows resolve relative to it. The ViT features are n_slots-independent, so this
# one cache is reused by every hier_router run / Optuna trial.
#
# NOTE: the val pilot (~28k ADE + ~1.1k PACO images) fits the 30-min slot. For the
# full PACO-train build (+~45k images) use --partition=gpu_a100_il (longer) or chunk.
#
#   sbatch submit_precompute_combined_square.sh

#SBATCH --job-name=combprecomp
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-combined_precompute_square.out
#SBATCH --error=slurm-%j-combined_precompute_square.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

CSV=FG-datset/parts_color_vqa_combined.csv
IMG_ROOT=FG-datset/ade20k/ADE20K_2021_17_01/images   # used for ADE rows; PACO rows are absolute
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
    --dino_cache_out FG-datset/dino_feat_cache_combined_square.pt \
    --text_cache_out FG-datset/text_feat_cache_combined_t5_spans.pt
