#!/bin/bash
# Step 8 — the thesis figures.
#
#   figures/hier_dinosaur_tree.png    input → object slots → part sub-slots (method chapter)
#   figures/router_trace.png          one routing trace, object → part → answer
#   figures/paco_localization*.png    PACO localisation rows
#   figures/cub_localization.png      CUB localisation rows
#   figures/candidates/               a pool of candidate rows to choose from
#
# The localisation figures are chosen by hand: run the pool first, look at the contact sheets,
# then pass the row indices you want through PACO_ROWS / CUB_ROWS (the defaults are the rows used
# in the thesis, which are only reproduced with the thesis checkpoints and the same seeds).
# Everything runs on CPU in a few minutes.
#
#   bash experiments/08_make_figures.sh              # figures + candidate pools
#   STAGE=pool bash experiments/08_make_figures.sh   # only the pools
set -euo pipefail
source "$(dirname "$0")/common.sh"

FIGURES=${FIGURES:-$ROOT/figures}
STAGE=${STAGE:-all}
POOL=${POOL:-200}
PACO_ROWS=${PACO_ROWS:-"0:146,10,114,140,197,196"}
CUB_ROWS=${CUB_ROWS:-"1:4,6;3:1,9;5:4,9"}
TREE_IMAGE=${TREE_IMAGE:-$ROOT/data/coco/val2017/000000039769.jpg}
paco=$CKPT_DIR/paco_hier_router/best_model.pt
cub=$CKPT_DIR/cub_hier_router/best_model.pt
mkdir -p "$FIGURES"

cmds=()
cmds+=("python -m hier_dinosaur.viz.figures --checkpoint $paco --n_samples $POOL --seed 0 --page_rows 20 \
    --colorbar --out_dir $FIGURES/candidates/paco --prefix paco")
cmds+=("python -m hier_dinosaur.viz.figures --checkpoint $cub --n_samples 12 --seed 1 \
    --colorbar --out_dir $FIGURES/candidates/cub --prefix cub")
if [ "$STAGE" = "all" ]; then
  cmds+=("python -m hier_dinosaur.viz.tree --image $TREE_IMAGE --n_slots 7 --children 3 --out $FIGURES/hier_dinosaur_tree.png")
  cmds+=("python -m hier_dinosaur.viz.routing --checkpoint $paco --n_samples 4 --seed 6 --out_dir $FIGURES/traces")
  cmds+=("python -m hier_dinosaur.viz.figures --checkpoint $paco --n_samples $POOL --seed 0 \
      --select '$PACO_ROWS' --colorbar --out_dir $FIGURES --prefix paco_localization_extra")
  cmds+=("python -m hier_dinosaur.viz.figures --checkpoint $cub --n_samples 12 \
      --select '$CUB_ROWS' --colorbar --out_dir $FIGURES --prefix cub_localization")
fi

printf -v joined '%s; ' "${cmds[@]}"
run_job figures "$PARTITION_CPU" "${TIME:-01:00:00}" 0 8 -- "$joined"
echo "  → $FIGURES"
