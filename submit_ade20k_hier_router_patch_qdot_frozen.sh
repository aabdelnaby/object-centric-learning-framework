#!/bin/bash
# hier_router PATCH_QDOT colour head on a FROZEN, PREVIOUSLY-TRAINED routing (built 2026-06-16).
# Like submit_ade20k_hier_router_patch_readout_frozen.sh (proven routing warm-started from job
# 4986866 + frozen, only the colour head trains) BUT the colour readout is the patch_qdot
# MECHANISM restricted to each child's patches:
#   * project the raw DINO patches (d_vit->d_slot) AND the "<y> <x>" query into a common space,
#   * an explicit query·patch DOT-PRODUCT scores every patch (no learned K/V, no multi-head),
#   * softmax, then CONFINE the attention to the patches that belong to the routed child node
#     (multiply by that child's slot-attention mask, renormalise) and pool,
#   * concat f_readout("<y> <x>") and read colour — done PER child, then path-marginalised.
#
# vs the plain patch-readout (--router_color_source patch, which pools patches by the slot mask
# alone): here a LEARNED query·patch dot-product reweights the patches WITHIN each child's region,
# so the colour head can focus on the colour-bearing patches of the part rather than averaging the
# whole child mask. Routing (P(j|y), P(k|j,x)) is the SAME proven, frozen routing as job 4986866.
#
# So the comparison ladder at FIXED proven routing is:
#   slot vector (job 4986866, val 0.5945)  vs  mask-pooled patches (patch frozen)  vs  qdot-within-
#   child patches (THIS run). Isolates whether a learned within-region patch selection beats both.
#
# IMPLEMENTATION: --router_color_source patch_qdot (builds ChildMaskedQDotColorHead; projects
# patches by default, --router_qdot_raw for raw d_vit); --router_init_ckpt warm-starts the routing
# (q_parent/q_child/child_mlp + text_projector), --router_freeze_routing freezes them so only the
# qdot colour head (~365k params) trains. Arch flags MUST match the source run (n_slots 9,
# recursive_children 5, child_scorer mlp, router_temp 0.95, recursive_spread 0.0).
#
# CACHE NOTE: text built in RAM (--text_in_memory); the square DINO patch cache (196 tokens) is
# loaded from disk — same cache the source run used, so the patches are identical features.

#SBATCH --job-name=adeQDfz
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ade20k_hier_router_patch_qdot_frozen.out
#SBATCH --error=slurm-%j-ade20k_hier_router_patch_qdot_frozen.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

POOLER_LAYERS=5

CSV=FG-datset/paco_questions.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Proven routing to warm-start + freeze (slot-readout run, val 0.5945).
INIT_CKPT=runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt
CKPT_DIR=runs/ade20k_hier_router_patch_qdot_frozen/${SLURM_JOB_ID:-local}

# Identical to submit_ade20k_hier_router_patch_readout_frozen.sh EXCEPT
# --router_color_source patch (mask-pool)  ->  patch_qdot (learned within-child dot-product).
python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --router_color_source patch_qdot \
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
