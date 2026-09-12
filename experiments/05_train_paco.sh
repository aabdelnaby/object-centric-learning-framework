#!/bin/bash
# Step 5 — train the five PACO models of Table 1.
#
#   hier_router               the full object→part router          Table 1 row 1, Tables 2-3, figures
#   hier_router_parent_only   ablation without the part level      Table 1 row 5
#   patch_qdot_projected      flat control, projected patches      Table 1 row 2, Table 2
#   patch_qdot_raw            flat control, raw patches            Table 1 row 3, Table 2
#   patch_qca                 flat control, cross-attention        Table 1 row 4
#
# Note for the two Patch-QDot models: the thesis runs dropped the first four patch tokens, having
# assumed the feature cache still held DINOv3's register tokens, which it does not. Runs from this
# branch use all 196 patches, so they land slightly above the published numbers (raw 0.587 vs 0.578,
# projected 0.606 vs 0.593) and their attention maps sit on the true grid. Thesis checkpoints still
# load and still reproduce their published numbers; see experiments/07_eval_tables.sh for how this
# affects grounding.
#
# Every model trains only its head on the cached features: AdamW 2e-4, weight decay 0.02,
# 10k warm-up steps then cosine, batch 128, NLL. All of them overfit within the budget, so the
# best-validation checkpoint is what gets reported. The patch models converge in about 15 minutes;
# the router uses the longer partition.
#
#   bash experiments/05_train_paco.sh                  # all five
#   bash experiments/05_train_paco.sh hier_router      # one
#   MAX_STEPS=300 bash experiments/05_train_paco.sh hier_router   # smoke test
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODELS=${*:-"hier_router hier_router_parent_only patch_qdot_projected patch_qdot_raw patch_qca"}
CKPT=${CKPT:-$ROOT/checkpoints/dinosaur_dinov3_vits16_coco.ckpt}
MAX_STEPS=${MAX_STEPS:-500000}
SEED=${SEED:-}
EXTRA=${EXTRA:-}
[ -n "$SEED" ] && EXTRA="$EXTRA --seed $SEED"

for model in $MODELS; do
  case $model in
    hier_router|hier_router_parent_only) part=$PARTITION_LONG; time=${TIME:-03:00:00}; args="--n_slots 9 --children 5" ;;
    patch_qdot_raw|patch_qdot_projected|patch_qca) part=$PARTITION_SHORT; time=${TIME:-00:30:00}; args="" ;;
    *) echo "unknown model: $model" >&2; exit 1 ;;
  esac
  out=$RUNS_ROOT/paco/$model
  run_job "paco_$model" "$part" "$time" 1 12 -- \
    "python -m hier_dinosaur.train --model $model --dataset paco --out $out \
       --dinosaur_ckpt $CKPT --max_steps $MAX_STEPS $args $EXTRA"
  echo "  → $out"
done
