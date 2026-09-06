#!/bin/bash
# Step 2 — build the PACO-LVIS part-colour question set (thesis Section 4.1).
#
#   questions + part masks  (CPU)            data_prep/paco/build_vqa_csv.py
#   part crops for the colour fallback (CPU) data_prep/paco/extract_part_crops.py
#   InternVL-3 colours for parts PACO left unlabelled (GPU array, internvl_env)
#   answers: PACO colour first, VLM fallback data_prep/paco/finalize_csv.py
#   train + val concatenated                 data_prep/paco/make_splits.py
#
# Result: data/paco/paco_questions.csv (50,679 train + 2,656 val questions, 12 colour classes)
# and data/paco/paco_val_masks.csv + part_masks/val/*.png used by the grounding evaluation.
#
# The steps are chained with SLURM dependencies, so this returns as soon as everything is queued.
# Needs the COCO images and the PACO annotations (docs/DATA.md).
#
#   bash experiments/02_build_paco_dataset.sh
#   STAGE=csv bash experiments/02_build_paco_dataset.sh    # only the first stage
set -euo pipefail
source "$(dirname "$0")/common.sh"

DATA=${DATA:-$ROOT/data/paco}
COCO=${COCO:-$ROOT/data/coco}
SHARDS=${SHARDS:-8}
INTERNVL_ENV=${INTERNVL_ENV:-internvl_env}
STAGE=${STAGE:-all}
jid() { awk '{print $NF}'; }

if [ "$STAGE" = "all" ] || [ "$STAGE" = "csv" ]; then
  CSV_JOB=$(run_job paco_csv "$PARTITION_CPU" 04:00:00 0 8 -- \
    "python data_prep/paco/build_vqa_csv.py --split train --coco-root $COCO --out-dir $DATA; \
     python data_prep/paco/build_vqa_csv.py --split val   --coco-root $COCO --out-dir $DATA" | jid)
  echo "questions + masks: job $CSV_JOB"
  [ "$STAGE" = "csv" ] && exit 0
fi

# Only the parts PACO left without a colour go to the VLM (about 18 % of them).
CROP_JOB=$(SBATCH_EXTRA="--dependency=afterok:$CSV_JOB" run_job paco_crops "$PARTITION_CPU" 02:00:00 0 16 -- \
  "python data_prep/paco/extract_part_crops.py --split train --data-dir $DATA --coco-root $COCO --only-unknown --workers 16; \
   python data_prep/paco/extract_part_crops.py --split val   --data-dir $DATA --coco-root $COCO --only-unknown --workers 16" | jid)
echo "part crops: job $CROP_JOB"

VLM_JOB=$(SBATCH_EXTRA="--dependency=afterok:$CROP_JOB --array=0-$((SHARDS - 1))" \
  CONDA_ENV=$INTERNVL_ENV run_job paco_vlm "$PARTITION_SHORT" 00:30:00 1 12 -- \
  "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
   python data_prep/paco/annotate_colors_internvl.py --split train --data-dir $DATA --only-unknown \
     --shard \$SLURM_ARRAY_TASK_ID --num-shards $SHARDS --skip-done; \
   python data_prep/paco/annotate_colors_internvl.py --split val --data-dir $DATA --only-unknown \
     --shard \$SLURM_ARRAY_TASK_ID --num-shards $SHARDS --skip-done" | jid)
echo "InternVL colours: array job $VLM_JOB ($SHARDS shards, $INTERNVL_ENV)"

FIN_JOB=$(SBATCH_EXTRA="--dependency=afterok:$VLM_JOB" run_job paco_finalize "$PARTITION_CPU" 00:30:00 0 4 -- \
  "python data_prep/paco/finalize_csv.py --split train --data-dir $DATA; \
   python data_prep/paco/finalize_csv.py --split val --data-dir $DATA; \
   python data_prep/paco/make_splits.py --train-csv $DATA/paco_train.csv --val-csv $DATA/paco_val.csv \
     --out $DATA/paco_questions.csv; \
   cp $DATA/paco_val_masks.csv $DATA/paco_val_masks.csv" | jid)
echo "answers + splits: job $FIN_JOB  →  $DATA/paco_questions.csv"
