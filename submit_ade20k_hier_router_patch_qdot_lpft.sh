#!/bin/bash
# hier_router PATCH_QDOT colour head — LP-FT (linear-probe → finetune) (built 2026-06-16).
# The principled finetune regime for "proven routing + new colour head":
#   * Stage 1 (LINEAR PROBE, already done) = the FROZEN qdot run (job 5040143): proven routing
#     held fixed, only the qdot colour head trained (best val 0.599).
#   * Stage 2 (FINETUNE, THIS run): init the WHOLE model — routing AND the trained qdot head —
#     from that linear-probe checkpoint (--router_init_ckpt + --router_init_color_head), then
#     finetune everything jointly at a LOW lr. Because the head is already trained, there is no
#     random-head-vs-pretrained-backbone tension (the failure mode of naive finetuning), so the
#     routing can co-adapt to the patch-qdot colour evidence without being destabilised.
#
# Regime (vs the naive finetune job 5040507, which warm-started routing only + random head at
# lr 2e-4 and overfit from ~epoch 11):
#   * lr 3e-5 (10x lower — best-val landed in the low-lr ramp, so the model wants low lr),
#   * warmup 200 (head already trained; no 10k protective ramp needed),
#   * max_steps 15000 (~37 epochs → the cosine schedule actually DECAYS over the run, instead of
#     sitting near peak as it did with max_steps 500000),
#   * weight_decay 0.05 + qdot head dropout 0.1 (curb the fast colour-head overfit),
#   * patience 15 (early-stop on best val).
# NOTE: label_smoothing is a no-op for the NLL router head, so it is omitted.
#
# Arch flags MUST match the LP/source run (n_slots 9, recursive_children 5, child_scorer mlp,
# router_temp 0.95, recursive_spread 0.0) so the warm-started tensors line up.
#
# CACHE NOTE: text built in RAM (--text_in_memory); square DINO patch cache (196 tokens) from disk.

#SBATCH --job-name=adeQDlpft
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ade20k_hier_router_patch_qdot_lpft.out
#SBATCH --error=slurm-%j-ade20k_hier_router_patch_qdot_lpft.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

POOLER_LAYERS=5

CSV=FG-datset/paco_questions.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Linear-probe checkpoint (frozen qdot head on proven routing, best val 0.599) — init routing+head.
INIT_CKPT=runs/ade20k_hier_router_patch_qdot_frozen/5040143/slots_9/best_model.pt
CKPT_DIR=runs/ade20k_hier_router_patch_qdot_lpft/${SLURM_JOB_ID:-local}

python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --router_color_source patch_qdot \
    --router_init_ckpt "$INIT_CKPT" \
    --router_init_color_head \
    --router_qdot_dropout 0.1 \
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
    --lr               3e-5 \
    --warmup_steps     200 \
    --max_steps        15000 \
    --patience         50 \
    --checkpoint_every 10 \
    --viz_n_samples 4 \
    --num_heads        64 \
    --skip_per_query_eval \
    --recursive_children 5 \
    --recursive_spread 0.0 \
    --pooler_dropout 0.0 \
    --router_temp 0.95 \
    --child_scorer mlp \
    --weight_decay 0.05 \
    --checkpoint_dir   "$CKPT_DIR"
