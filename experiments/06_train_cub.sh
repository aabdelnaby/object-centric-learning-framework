#!/bin/bash
# Step 6 — train the HierRouter on CUB-200 colour questions (thesis Section 4.5).
#
# Same router and recipe as on PACO, with 7 object slots (bird plus background) and the attribute
# carried on the readout channel so "back colour" and "back pattern" can be told apart. Per-part
# accuracies are printed and appended to per_query_stats.csv every 30 epochs; the thesis per-part
# table is the epoch-30 block, the headline 56.7 % is the best-test checkpoint.
#
#   bash experiments/06_train_cub.sh
#   MAX_STEPS=300 bash experiments/06_train_cub.sh   # smoke test
set -euo pipefail
source "$(dirname "$0")/common.sh"

CKPT=${CKPT:-$ROOT/checkpoints/dinosaur_dinov3_vits16_coco.ckpt}
MAX_STEPS=${MAX_STEPS:-500000}
OUT=${OUT:-$RUNS_ROOT/cub/hier_router}

run_job cub_hier_router "$PARTITION_LONG" "${TIME:-02:00:00}" 1 12 -- \
  "python -m hier_dinosaur.train --model hier_router --dataset cub --out $OUT \
     --dinosaur_ckpt $CKPT --n_slots 7 --children 5 --max_steps $MAX_STEPS ${EXTRA:-}"
echo "  → $OUT"
