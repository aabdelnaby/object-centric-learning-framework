#!/bin/bash
# Train the CUB attribute classifier on FT-DINOSAUR slots.
#
# Uses --feat_cache: the ftdinosaur ViT-B/14 encoder features are precomputed
# (ftdino_feat_cache.pt via submit_precompute_ftdinosaur.sh), so each step only
# runs the cheap slot attention + gated cross-attn + head. That means a large
# batch (256) is fine and one epoch over the ~137k train / ~133k val rows takes
# a couple of minutes. (At batch 4 a single epoch did not finish in the 30-min
# wall limit, so no per-epoch line ever printed — that was the earlier hang.)
#
# Submit:
#   sbatch submit_cub_ftdinosaur.sh
#
# Resumable: if killed at wall-time, re-submitting picks up from last.pt thanks
# to --resume (trainable weights + optimizer + LR schedule + epoch counter).
#
# Results land in:
#   runs/cub_ftdinosaur/slots_7/best_model.pt
#   runs/cub_ftdinosaur/slots_7/metrics.csv
#   runs/cub_ftdinosaur/slots_7/viz/epoch_*.png

#SBATCH --job-name=cub_ftdino
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-cub_ftdino.out
#SBATCH --error=slurm-%j-cub_ftdino.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# First run downloads the ftdinosaur checkpoint (~381 MB) to
# ~/.cache/torch/hub/checkpoints/. Compute nodes without internet should warm
# this cache from a login node once (the model auto-downloads on build).

python train.py \
    --dataset           cub \
    --slot_backend      ftdinosaur \
    --ftdinosaur_model  dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k \
    --n_slots           7 \
    --max_epochs        50 \
    --batch_size        256 \
    --lr                1e-4 \
    --weight_decay      1e-2 \
    --warmup_steps      200 \
    --patience          20 \
    --checkpoint_every  10 \
    --checkpoint_dir    runs/cub_ftdinosaur \
    --feat_cache \
    --pooler gated_attn 