#!/bin/bash
# Precompute the ftdinosaur encoder-feature cache for CUB.
#
# Caches the frozen ViT-B/14 encoder features (256x768 per image) for every
# unique CUB image. Slot attention is NOT cached — it is re-run live at train
# time (cheap, and keeps the random slot init fresh each epoch). This removes
# the expensive ViT-B/14 forward from every training step, so you can raise the
# batch size back up and drop --feat_cache-less slowness.
#
# The RoBERTa text cache (text_feat_cache.pt, 28 queries) is backend-independent
# and already exists, so we --skip_text and reuse it.
#
# Submit:
#   sbatch submit_precompute_ftdinosaur.sh
#
# Output (~4.5 GB fp16):
#   FG-datset/CUB_200_2011/ftdino_feat_cache.pt
#
# Then train with the cache:
#   python train.py --slot_backend ftdinosaur --feat_cache --n_slots 7 ...
# (train.py auto-points --dino_cache at ftdino_feat_cache.pt for this backend.)

#SBATCH --job-name=ftdino_precompute
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-ftdino_precompute.out
#SBATCH --error=slurm-%j-ftdino_precompute.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# First run downloads the ftdinosaur checkpoint (~381 MB) to
# ~/.cache/torch/hub/checkpoints/ (already cached if you ran the trainer once).

python precompute_features.py \
    --slot_backend  ftdinosaur \
    --ftdinosaur_model dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k \
    --csv           FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv \
    --image_root    FG-datset/CUB_200_2011/images \
    --batch_size    64 \
    --num_workers   12 \
    --fp16 \
    --skip_text
