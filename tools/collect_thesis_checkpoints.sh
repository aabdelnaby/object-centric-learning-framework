#!/bin/bash
# Copy the six trained heads reported in the thesis (plus their training logs) from the
# original research checkout into checkpoints/thesis/<name>/ (gitignored).
#
#   OLD_REPO=/path/to/original/checkout bash tools/collect_thesis_checkpoints.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OLD=${OLD_REPO:-$ROOT/../object-centric-learning-framework}
DEST=$ROOT/checkpoints/thesis

declare -A SRC=(
  [paco_hier_router]="runs/ade20k_hier_router_readout/4986866/slots_9"          # Table 1 row 1, Tables 2-3, PACO figures
  [paco_hier_router_parent_only]="runs/ade20k_hier_router_parent_only/4985890/slots_9"  # Table 1 row 5
  [paco_patch_qdot_raw]="runs/ade20k_patch_qdot_raw/5030504/patches"            # Table 1 row 3, Table 2
  [paco_patch_qdot_projected]="runs/ade20k_patch_qdot_projected/5030505/patches"  # Table 1 row 2, Table 2
  [paco_patch_qca]="runs/ade20k_qca_patch/4995863/patches"                      # Table 1 row 4
  [cub_hier_router]="runs/cub_hier_router/5043374/slots_7"                      # CUB section
)
for name in "${!SRC[@]}"; do
  src="$OLD/${SRC[$name]}"
  mkdir -p "$DEST/$name"
  cp "$src/best_model.pt" "$DEST/$name/best_model.pt"
  for f in metrics.csv per_query_stats.csv label_vocab.json; do
    [ -f "$src/$f" ] && cp "$src/$f" "$DEST/$name/$f"
  done
  printf '  %-30s <- %s\n' "$name" "$src"
done
echo "done → $DEST"
