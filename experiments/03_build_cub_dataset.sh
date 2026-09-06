#!/bin/bash
# Step 3 — build the CUB-200 part-attribute question set (thesis Section 4.5).
#
# Turns the CUB attribute annotations into "What is the <part> <attribute> of the bird?" rows,
# dense-ranked by annotator certainty. Training uses the 16 colour questions (15 colour classes,
# selected at run time with category_filter=color) on the official train/test split.
# Needs the extracted CUB_200_2011 release (docs/DATA.md). Runs in a couple of minutes on a CPU.
#
#   bash experiments/03_build_cub_dataset.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

CUB_ROOT=${CUB_ROOT:-$ROOT/data/cub/CUB_200_2011}
OUT=${OUT:-$ROOT/data/cub/cub200_questions.csv}

run_job cub_csv "$PARTITION_CPU" 00:30:00 0 4 -- \
  "python data_prep/cub/build_cub_csv.py --cub-root $CUB_ROOT --out $OUT"
echo "→ $OUT"
