#!/bin/bash
# hier_router with the QUERY-CONDITIONED COLOUR READOUT (built 2026-06-10).
# One change from submit_ade20k_hier_router_square.sh: the colour head reads
#   P(colour | child_slot, f_readout("<y> <x>"))   e.g. f_readout("car door")
# instead of P(colour | child_slot). The routed child slot localises the part;
# the "<y> <x>" compound noun (object then part) tells the shared head WHICH
# entity's colour to report. f_readout is a learned Linear(d_slot, d_slot) whose
# output is concatenated with the slot (colour_head input = 2*d_slot). Readout
# conditioning is ON by default for pooler=hier_router (disable: --router_no_readout).
#
# This is the apples-to-apples partner of the square run (job 4976867 = no-readout,
# identical otherwise): the val-acc gap isolates the value of conditioning the
# colour readout on the query.
#
# CACHE NOTE: the on-disk text cache FG-datset/text_feat_cache_combined_t5_spans.pt
# predates the readout channel (no readout_vec → would silently fall back to the
# part vec). So we build the text features in RAM with --text_in_memory (T5 over the
# ~1,117 unique questions at startup; this now also encodes the "<y> <x>" readout
# noun). The square DINO image cache is unaffected and still loaded from disk.

#SBATCH --job-name=adeRDO
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j-ade20k_hier_router_readout.out
#SBATCH --error=slurm-%j-ade20k_hier_router_readout.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# T-n downstream depth; TF-2 is fast and sufficient.
POOLER_LAYERS=5

CSV=FG-datset/coco_paco/parts_color_vqa_paco_trainval_gt.csv
# COCO-trained plain-DINO ViT-S/16 @224 (384-dim) backbone — natural-image / in-domain.
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Per-job checkpoint dir (SLURM_JOB_ID; falls back to "local" outside SLURM).
CKPT_DIR=runs/ade20k_hier_router_readout/${SLURM_JOB_ID:-local}

# ── Train + validate (per-epoch val accuracy is the signal) ──────────────────
# Text (T5, with the "<y> <x>" readout noun) is computed in RAM at startup
# (--text_in_memory); the square DINO image cache is loaded from disk.
python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --rank_method      attribution \
    --pooler_layers    "$POOLER_LAYERS" \
    --n_slots          7 \
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
    --recursive_children 4 \
    --recursive_spread 0.0 \
    --pooler_dropout 0.0 \
    --router_temp 0.95 \
    --child_scorer mlp \
    --weight_decay 0.02 \
    --checkpoint_dir   "$CKPT_DIR"
