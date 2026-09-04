#!/bin/bash
# hier_router PATCH colour head on a FROZEN, PREVIOUSLY-TRAINED routing (built 2026-06-16).
# Like submit_ade20k_hier_router_patch_readout.sh (--router_color_source patch) but the
# routing is NOT trained from scratch: it is warm-started from the proven slot-readout run
# (job 4986866, val 0.5945) and then FROZEN, so ONLY the new patch-sourced colour head
# (color_head + f_readout, ~233k params) is learned. P(j|y), P(k|j,x) and the text
# projector are held at their trained values.
#
# Why: the from-scratch patch-readout run (job 5039798) could learn a DIFFERENT routing,
# confounding "patches help the colour readout" with "routing changed". This run removes
# that confound — identical proven routing decisions, the ONLY difference vs job 4986866 is
# that the colour head reads the raw DINO patches each child grounds to (z_jk = Σ_n a_jkn·
# patch_n, the child slot-attn mask pooling frozen ViT features) instead of the child SLOT
# vector. So the val-acc delta vs 0.5945 is purely "colour from grounded patches vs from the
# slot vector", at fixed routing. Tests whether the slot vector was the colour bottleneck.
#
# IMPLEMENTATION: --router_init_ckpt <best_model.pt> warm-starts text_projector + q_parent/
# q_child/child_mlp (colour head stays fresh); --router_freeze_routing freezes those so the
# optimiser only sees the colour head. Architecture flags MUST match the source run so the
# routing tensors line up (n_slots 9, recursive_children 5, child_scorer mlp, router_temp
# 0.95, recursive_spread 0.0) — they do (clone of the readout submit script).
#
# CACHE NOTE: text built in RAM (--text_in_memory); the square DINO patch cache (196 tokens)
# is loaded from disk — same cache the source run used, so pooled patches are identical.

#SBATCH --job-name=adePATfz
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ade20k_hier_router_patch_readout_frozen.out
#SBATCH --error=slurm-%j-ade20k_hier_router_patch_readout_frozen.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

POOLER_LAYERS=5

CSV=FG-datset/paco_questions.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Proven routing to warm-start + freeze (slot-readout run, val 0.5945).
INIT_CKPT=runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt
CKPT_DIR=runs/ade20k_hier_router_patch_readout_frozen/${SLURM_JOB_ID:-local}

# Identical to submit_ade20k_hier_router_readout.sh EXCEPT: --router_color_source patch,
# --router_init_ckpt (warm-start routing), --router_freeze_routing (train colour head only).
python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --router_color_source patch \
    --router_init_ckpt "$INIT_CKPT" \
    --router_freeze_routing \
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
