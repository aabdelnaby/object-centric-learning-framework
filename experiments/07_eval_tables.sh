#!/bin/bash
# Step 7 — the three PACO tables and the CUB numbers.
#
#   Table 1  colour accuracy (top-1/2/3) of the five PACO models
#   Table 3  marginal vs MAP-path readout of the full router (part of its Table 1 pass)
#   Table 2  grounding faithfulness of the router and the two Patch-QDot models
#   CUB      overall accuracy and the per-part breakdown
#
# Results are written as JSON to $RESULTS and printed as markdown tables by the last step.
# By default it evaluates the trained heads in checkpoints/thesis (tools/collect_thesis_checkpoints.sh);
# point CKPT_DIR at runs/ to evaluate your own training runs instead.
#
#   bash experiments/07_eval_tables.sh
#   CKPT_DIR=$PWD/runs bash experiments/07_eval_tables.sh   # after steps 5 and 6
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$RESULTS"

find_ckpt() {  # accept both checkpoints/thesis/<name>/ and runs/<dataset>/<model>/ layouts
  for p in "$CKPT_DIR/$1/best_model.pt" "$CKPT_DIR/${1/_//}/best_model.pt"; do
    [ -f "$p" ] && { echo "$p"; return; }
  done
  echo "missing checkpoint for $1 (looked under $CKPT_DIR)" >&2
  return 1
}

PACO_MODELS=${PACO_MODELS:-"hier_router hier_router_parent_only patch_qdot_projected patch_qdot_raw patch_qca"}
SEED=${SEED:-0}
cmds=()
for m in $PACO_MODELS; do
  ckpt=$(find_ckpt "paco_$m")
  cmds+=("python -m hier_dinosaur.evaluate --checkpoint $ckpt --seed $SEED \
      --out $RESULTS/paco_$m.json --predictions $RESULTS/paco_${m}_predictions.csv")
done
cub_ckpt=$(find_ckpt cub_hier_router) || true
if [ -n "${cub_ckpt:-}" ]; then
  cmds+=("python -m hier_dinosaur.evaluate --checkpoint $cub_ckpt --seed $SEED --per_query --out $RESULTS/cub_hier_router.json")
  stats=$(dirname "$cub_ckpt")/per_query_stats.csv
  [ -f "$stats" ] && cmds+=("python -m hier_dinosaur.evaluate --from_stats $stats --epoch ${CUB_EPOCH:-30} --out $RESULTS/cub_per_part_epoch${CUB_EPOCH:-30}.json")
fi
cmds+=("python -m hier_dinosaur.grounding --router_ckpt $(find_ckpt paco_hier_router) \
    --patch_ckpt $(find_ckpt paco_patch_qdot_projected) --patch_ckpt $(find_ckpt paco_patch_qdot_raw) \
    --n ${N_GROUNDING:-800} --seed $SEED --masks_csv data/paco/paco_val_masks.csv --image_root data/coco \
    --out $RESULTS/paco_grounding.json --per_sample_csv $RESULTS/paco_grounding_per_sample.csv")
cmds+=("python -m hier_dinosaur.evaluate --summarize $RESULTS")

printf -v joined '%s; ' "${cmds[@]}"
run_job eval_tables "$PARTITION_SHORT" "${TIME:-00:30:00}" 1 12 -- "$joined"
echo "  → $RESULTS  (tables are printed at the end of the job log)"
