#!/bin/bash
# Step 4 — cache the frozen DINOv3 patch features for every image of both datasets.
#
# All heads train on these features, so the ViT runs once per image instead of once per step:
#   data/caches/dino_paco_square224.pt   ~22k COCO images   (~8 GB fp16, about 15 min on an A100)
#   data/caches/dino_cub_square224.pt    ~11.8k CUB images  (~1.7 GB fp16, about 5 min)
# 196 tokens per image at 224 px with square resize (the preprocessing the slot module was trained with).
#
#   bash experiments/04_precompute_features.sh            # both datasets
#   DATASET=cub bash experiments/04_precompute_features.sh
#   LIMIT=200 bash experiments/04_precompute_features.sh   # smoke test on a few images
set -euo pipefail
source "$(dirname "$0")/common.sh"

DATASET=${DATASET:-both}
CKPT=${CKPT:-$ROOT/checkpoints/dinosaur_dinov3_vits16_coco.ckpt}
LIMIT=${LIMIT:-}
EXTRA=${FORCE:+--force}
[ -n "$LIMIT" ] && EXTRA="$EXTRA --limit $LIMIT"

if [ "$DATASET" = "both" ] || [ "$DATASET" = "paco" ]; then
  run_job precompute_paco "$PARTITION_SHORT" 00:30:00 1 12 -- \
    "python -m hier_dinosaur.features --csv data/paco/paco_questions.csv --image_root data/coco \
       --dinosaur_ckpt $CKPT --out data/caches/dino_paco_square224.pt --batch_size 64 --num_workers 12 $EXTRA"
fi
if [ "$DATASET" = "both" ] || [ "$DATASET" = "cub" ]; then
  run_job precompute_cub "$PARTITION_SHORT" 00:30:00 1 12 -- \
    "python -m hier_dinosaur.features --csv data/cub/cub200_questions.csv \
       --image_root data/cub/CUB_200_2011/images \
       --dinosaur_ckpt $CKPT --out data/caches/dino_cub_square224.pt --batch_size 64 --num_workers 12 $EXTRA"
fi
