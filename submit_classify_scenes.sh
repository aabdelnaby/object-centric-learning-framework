#!/bin/bash
# Zero-shot scene classification of every unique image in paco_questions.csv with
# CLIP (open_clip ViT-L-14, laion2b). Produces FG-datset/paco_image_scenes.csv:
# one row per image with the predicted scene, confidence, top-3, and a COCO-caption
# cross-check. Used to define an OOD held-out scene for hier_router vs baseline.
#
# NOTE: ViT-L-14 weights must already be cached on disk (pre-downloaded on the
# login node) because compute nodes have no internet.
#SBATCH --job-name=sceneCLF
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-classify_scenes.out
#SBATCH --error=slurm-%j-classify_scenes.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

python classify_scenes_clip.py \
  --csv FG-datset/paco_questions.csv \
  --out FG-datset/paco_image_scenes.csv \
  --model ViT-L-14 \
  --pretrained laion2b_s32b_b82k \
  --batch_size 256
