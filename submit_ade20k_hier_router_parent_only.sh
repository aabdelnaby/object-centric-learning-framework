#!/bin/bash
# ABLATION: parent-only hier_router on ADE20K part-COLOR VQA.
#
# Identical to submit_ade20k_hier_router_square.sh (the full parent→child model,
# job 4972061) EXCEPT for the single flag --router_parent_only. That collapses the
# head to
#       P(a) = Σ_j P(j|y) · P(a | s_j)
# i.e. route only over object slots and read the colour straight off the chosen
# parent — NO child refinement, and the part word <x> is unused. Comparing this
# run's val acc against the full model isolates the utility of the child level.
#
# Everything else is held fixed for a fair comparison: same n_slots, lr, router_temp,
# weight_decay, square resize, DINOv3 backbone, T5 text. --child_scorer / --recursive_children
# below are INERT in parent-only mode (no children are built) — kept only so this is a
# literal one-line diff from the full run.
#
#   sbatch submit_ade20k_hier_router_parent_only.sh

#SBATCH --job-name=adePAR
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ade20k_vqa_parent_only.out
#SBATCH --error=slurm-%j-ade20k_vqa_parent_only.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

POOLER_LAYERS=5

CSV=FG-datset/paco_questions.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
CKPT_DIR=runs/ade20k_hier_router_parent_only/${SLURM_JOB_ID:-local}

python train.py \
    --dataset          ade20k \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --router_parent_only \
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
    --max_steps        60000 \
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
