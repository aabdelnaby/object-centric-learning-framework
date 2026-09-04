#!/bin/bash
# hier_router with PATCH-SOURCED COLOUR READOUT (built 2026-06-16).
# One change from submit_ade20k_hier_router_readout.sh: --router_color_source patch.
# The structured parent->child routing is UNCHANGED — P(j|y) and P(k|j,x) still decide
# *which* child slot to attend — but the per-child colour head no longer reads the
# abstracted child SLOT VECTOR c_jk. Instead it reads the raw DINO patches that child
# grounds to: z_jk = Σ_n a_jkn · patch_n (the child's slot-attention mask, renormalised,
# pooling the frozen ViT patch features), concatenated with f_readout("<y> <x>"). Colour
# head input is therefore d_vit(384)+d_slot(256)=640 instead of 2*d_slot=512.
#
# Hypothesis (from the grounding/accuracy dissociation, 2026-06-16): the router grounds
# the named part ~2.4x better than flat patches but its colour accuracy is only tied,
# because the slot VECTOR abstracts away low-level colour the patches retain. Routing to
# ground, then reading colour off the grounded patches, should keep the grounding AND
# recover patch-level colour — potentially beating both the slot-readout router (val
# 0.5945, job 4986866) and Patch-QDot.
#
# Apples-to-apples partner of the slot-readout run (job 4986866): IDENTICAL frozen DINOv3
# features / combined_square cache / PACO CSV / n_slots / recursive tree / router temp /
# child scorer / "<y> <x>" query / NLL loss / optimiser budget. ONLY the colour-evidence
# source differs, so the val-acc gap isolates "colour from grounded patches" vs "colour
# from slot vector". Separate weights, trained independently.
#
# CACHE NOTE: text built in RAM (--text_in_memory) for the "<y> <x>" readout noun; the
# square DINO patch cache (196 tokens) is loaded from disk — the same cache the slot-
# readout / qca / qdot runs use, so the pooled patches are bit-identical features.

#SBATCH --job-name=adePAT
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j-ade20k_hier_router_patch_readout.out
#SBATCH --error=slurm-%j-ade20k_hier_router_patch_readout.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# T-n downstream depth; TF-2 is fast and sufficient.
POOLER_LAYERS=5

CSV=FG-datset/paco_questions.csv
# COCO-trained plain-DINO ViT-S/16 @224 (384-dim) backbone — natural-image / in-domain.
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Per-job checkpoint dir (SLURM_JOB_ID; falls back to "local" outside SLURM).
CKPT_DIR=runs/ade20k_hier_router_patch_readout/${SLURM_JOB_ID:-local}

# ── Train + validate (per-epoch val accuracy is the signal) ──────────────────
# Identical to submit_ade20k_hier_router_readout.sh EXCEPT --router_color_source patch.
python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --router_color_source patch \
    --rank_method      attribution \
    --pooler_layers    "$POOLER_LAYERS" \
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
    --viz_n_samples 4 \
    --num_heads        64 \
    --skip_per_query_eval \
    --recursive_children 5 \
    --recursive_spread 0.0 \
    --pooler_dropout 0.0 \
    --router_temp 0.95 \
    --child_scorer mlp \
    --weight_decay 0.02 \
    --checkpoint_dir   "$CKPT_DIR"
