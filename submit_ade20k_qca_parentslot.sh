#!/bin/bash
# ParentSlot-QCA — flat (unstructured) CONTROL for the hier_router (baseline B1).
# The "<y> <x>" compound query ("car door", span ch3) cross-attends ONCE over the
# object/parent slots (empty slots masked, same 2% rule as the router); the single
# attended vector is read out with the SAME query-conditioned colour head
#   P(colour) = softmax( color_head([ z ; f_readout("<y> <x>") ]) )
# There is NO parent->child tree, NO P(j|y)/P(k|j,x), and NO path marginalisation.
# It isolates whether hier_router's gains come from the structured routing or merely
# from the compound query + query-conditioned readout pulling the right slot.
#
# Apples-to-apples partner of the readout hier_router run (job 4986866, val 0.5945):
# IDENTICAL frozen DINOv3 backbone, combined_square DINO cache, PACO CSV/split, "<y> <x>"
# query, readout-head shape, NLL loss, and optimiser/step budget. Only the head differs
# (flat cross-attention vs. structured routing). Separate weights, trained independently.
#
# CACHE NOTE: the on-disk text cache predates the readout/span channels, so we build the
# 4-channel text features in RAM with --text_in_memory (T5; this encodes the "<y> <x>"
# noun into span ch3). The square DINO image cache is loaded from disk.

#SBATCH --job-name=qcaPS
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ade20k_qca_parentslot.out
#SBATCH --error=slurm-%j-ade20k_qca_parentslot.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

CSV=FG-datset/paco_questions.csv
# COCO-trained DINOv3 ViT-S/16 @224 (384-dim) backbone — natural-image / in-domain.
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Per-job checkpoint dir (SLURM_JOB_ID; falls back to "local" outside SLURM).
CKPT_DIR=runs/ade20k_qca_parentslot/${SLURM_JOB_ID:-local}

# ── Train + validate (per-epoch val accuracy is the signal) ──────────────────
# Text (T5, with the "<y> <x>" span) is computed in RAM at startup (--text_in_memory);
# the square DINO image cache is loaded from disk. No --recursive_infer (qca has no tree).
python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           qca \
    --n_slots          9 \
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
