#!/bin/bash
# Link datasets, feature caches and the DINOSAUR checkpoint that already exist on this machine
# (in the original research checkout) into the layout this branch expects. Everything created
# here lives under data/ and checkpoints/, both gitignored. See docs/DATA.md for the layout and
# for how to obtain the files on a fresh machine.
#
#   OLD_REPO=/path/to/original/checkout bash tools/link_local_data.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OLD=${OLD_REPO:-$ROOT/../object-centric-learning-framework}
if [ ! -d "$OLD" ]; then
  echo "original checkout not found at $OLD (set OLD_REPO)" >&2
  exit 1
fi

link() {  # link <target> <link path>
  mkdir -p "$(dirname "$2")"
  ln -sfn "$1" "$2"
  printf '  %-45s -> %s\n' "$2" "$1"
}

cd "$ROOT"
echo "data/"
link "$OLD/scripts/datasets/data/coco"                                   data/coco
link "$OLD/scripts/datasets/outputs"                                     data/webdataset
link "$OLD/FG-datset/coco_paco/paco_annotations"                         data/paco/paco_annotations
link "$OLD/FG-datset/paco_questions.csv"                                 data/paco/paco_questions.csv
link "$OLD/FG-datset/coco_paco/parts_color_vqa_paco_val_masks.csv"       data/paco/paco_val_masks.csv
link "$OLD/FG-datset/coco_paco/part_masks"                               data/paco/part_masks
link "$OLD/FG-datset/CUB_200_2011"                                       data/cub/CUB_200_2011
link "$OLD/FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv" data/cub/cub200_questions.csv
link "$OLD/FG-datset/dino_feat_cache_combined_square.pt"                 data/caches/dino_paco_square224.pt
link "$OLD/FG-datset/CUB_200_2011/dino_feat_cache_dinov3_square.pt"      data/caches/dino_cub_square224.pt
echo "checkpoints/"
link "$OLD/checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt" checkpoints/dinosaur_dinov3_vits16_coco.ckpt
echo "done"
