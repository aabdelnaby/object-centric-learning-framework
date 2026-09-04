#!/bin/bash
# Patch-QCA — flat (unstructured) CONTROL for the hier_router (baseline B2).
# Identical to ParentSlot-QCA EXCEPT the visual tokens are the raw frozen DINOv3 patch
# tokens (200 = 4 register + 196 patch, projected d_vit->d_slot) instead of slots. The
# "<y> <x>" query ("car door", span ch3) cross-attends ONCE over those patch tokens and
# the attended vector is read out with the SAME query-conditioned colour head as the
# router / ParentSlot-QCA. No slots, no tree, no marginalisation.
# Tests: do we need object-centric slots at all, or are raw patch features enough?
#
# Apples-to-apples partner of the readout hier_router run (job 4986866, val 0.5945) and
# of ParentSlot-QCA: IDENTICAL frozen DINOv3 features, combined_square cache, PACO
# CSV/split, "<y> <x>" query, readout-head shape, NLL loss, optimiser/step budget.
# Uses PatchClassifier (--patch_control). Separate weights, trained independently.
#
# CACHE NOTE: text built in RAM (--text_in_memory) for the "<y> <x>" span ch3; the square
# DINO patch cache is loaded from disk (same cache the slot runs use).

#SBATCH --job-name=qcaPatch
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ade20k_qca_patch.out
#SBATCH --error=slurm-%j-ade20k_qca_patch.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

CSV=FG-datset/paco_questions.csv
# COCO-trained DINOv3 ViT-S/16 @224 (384-dim) backbone — natural-image / in-domain.
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Per-job checkpoint dir (SLURM_JOB_ID; falls back to "local" outside SLURM).
CKPT_DIR=runs/ade20k_qca_patch/${SLURM_JOB_ID:-local}

# ── Train + validate (per-epoch val accuracy is the signal) ──────────────────
# --patch_control → PatchClassifier (raw ViT patch tokens, no slots). d_vit defaults to
# 384 (CONFIG), matching the DINOv3-small feature dim. Text built in RAM; cache from disk.
python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           qca \
    --patch_control \
    --dinosaur_cfg     "$DINO_CFG" \
    --dinosaur_ckpt    "$DINO_CKPT" \
    --img_size         224 \
    --resize_mode      square \
    --feat_cache \
    --dino_cache     FG-datset/dino_feat_cache_combined_square.pt \
    --text_in_memory \
    --batch_size       128 \
    --lr               2e-4 \
    --warmup_steps     10000 \
    --max_steps        500000 \
    --patience         1000000 \
    --checkpoint_every 25 \
    --num_heads        8 \
    --skip_per_query_eval \
    --weight_decay 0.02 \
    --checkpoint_dir   "$CKPT_DIR"
