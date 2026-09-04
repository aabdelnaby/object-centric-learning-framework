#!/bin/bash
# OOD-bedroom experiment — BASELINE (Patch-QDot RAW, the flat control).
# Clone of submit_ade20k_patch_qdot_raw.sh, with two changes for the scene-OOD test:
#   1. trains on paco_questions_no_bedroom.csv (bedroom scene held out of train+val);
#   2. after training, evals best_model.pt on paco_questions_bedroom_only.csv (the
#      unseen scene) via eval_ood.py.
# Apples-to-apples partner: submit_ood_bedroom_hier_router.sh (same CSVs, same cache).

#SBATCH --job-name=oodBaseBed
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ood_bedroom_baseline.out
#SBATCH --error=slurm-%j-ood_bedroom_baseline.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

TRAIN_CSV=FG-datset/paco_questions_no_bedroom.csv
OOD_CSV=FG-datset/paco_questions_bedroom_only.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
DINO_CACHE=FG-datset/dino_feat_cache_combined_square.pt
CKPT_DIR=runs/ood_bedroom_patch_qdot_raw/${SLURM_JOB_ID:-local}

# ── Train + in-domain val (bedroom NOT in this CSV) → best_model.pt ───────────
python train.py \
    --dataset          ade20k \
    --csv_path         "$TRAIN_CSV" \
    --text_encoder     t5 \
    --pooler           patch_qdot \
    --patch_control \
    --dinosaur_cfg     "$DINO_CFG" \
    --dinosaur_ckpt    "$DINO_CKPT" \
    --img_size         224 \
    --resize_mode      square \
    --feat_cache \
    --dino_cache       "$DINO_CACHE" \
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

# ── OOD eval: best checkpoint on the held-out bedroom scene ───────────────────
BEST=$CKPT_DIR/patches/best_model.pt
echo "=== OOD eval on bedroom: $BEST ==="
python eval_ood.py \
    --checkpoint "$BEST" \
    --csv_path   "$OOD_CSV" \
    --split      val \
    --dino_cache "$DINO_CACHE"
