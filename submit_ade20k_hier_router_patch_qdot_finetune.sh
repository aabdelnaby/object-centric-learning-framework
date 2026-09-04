#!/bin/bash
# hier_router PATCH_QDOT colour head, proven routing WARM-STARTED then FINETUNED (built 2026-06-16).
# Same colour head as submit_ade20k_hier_router_patch_qdot_frozen.sh (ChildMaskedQDotColorHead:
# project patches + "<y> <x>" query, query·patch dot-product, CONFINE to each child's patches,
# pool, read out — per child, path-marginalised) and the same warm-start of the proven routing
# (job 4986866) — but WITHOUT --router_freeze_routing, so the routing is NOT held fixed: it starts
# from the proven q_parent/q_child/child_mlp + text_projector and is FINETUNED jointly with the
# fresh qdot colour head.
#
# vs the FROZEN qdot run (job 5040143): there the routing is fixed at the proven values, so the
# colour head must work with the exact existing parent/child decisions. Here the routing can ADAPT
# to the new patch-qdot colour evidence — e.g. shift which child a part routes to so the qdot head
# has better patches to read. The gap between the two isolates "does letting the routing co-adapt
# to the patch colour head help, beyond just swapping the readout?".
#
# Everything trains (routing ~592k + qdot head ~365k). Warm-start gives the routing a good init so
# it converges far faster than the from-scratch patch runs. Arch flags MUST match the source run
# (n_slots 9, recursive_children 5, child_scorer mlp, router_temp 0.95, recursive_spread 0.0) so the
# warm-started routing tensors line up.
#
# CACHE NOTE: text built in RAM (--text_in_memory); the square DINO patch cache (196 tokens) is
# loaded from disk — same cache the source run used.

#SBATCH --job-name=adeQDft
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ade20k_hier_router_patch_qdot_finetune.out
#SBATCH --error=slurm-%j-ade20k_hier_router_patch_qdot_finetune.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

POOLER_LAYERS=5

CSV=FG-datset/paco_questions.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Proven routing to warm-start (slot-readout run, val 0.5945); NOT frozen → finetuned.
INIT_CKPT=runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt
CKPT_DIR=runs/ade20k_hier_router_patch_qdot_finetune/${SLURM_JOB_ID:-local}

# Identical to submit_ade20k_hier_router_patch_qdot_frozen.sh EXCEPT: NO --router_freeze_routing
# (the warm-started routing is finetuned jointly with the qdot colour head).
python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --router_color_source patch_qdot \
    --router_init_ckpt "$INIT_CKPT" \
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
