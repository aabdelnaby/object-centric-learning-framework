#!/bin/bash
# Step 1 — pretrain the DINOSAUR slot module on COCO (the frozen backbone of every experiment).
#
# Frozen DINOv3 ViT-S/16 features → RandomConditioning → SlotAttention → AutoregressivePatchDecoder,
# trained for feature reconstruction (Adam 4e-4, 500k steps, batch 16). This is the only step that
# uses the vendored OCL framework; everything downstream loads the resulting checkpoint frozen.
#
# Needs the COCO webdataset shards (docs/DATA.md). Takes about two days on one A100; the thesis
# checkpoint came from the 11-slot run of this exact config.
#
#   bash experiments/01_train_dinosaur.sh              # 11 slots, full 500k steps
#   N_SLOTS=7 bash experiments/01_train_dinosaur.sh    # a different slot count
#   MAX_STEPS=200 bash experiments/01_train_dinosaur.sh   # smoke test
set -euo pipefail
source "$(dirname "$0")/common.sh"

N_SLOTS=${N_SLOTS:-11}
MAX_STEPS=${MAX_STEPS:-500000}
OUT=${OUT:-$RUNS_ROOT/dinosaur/slots_$N_SLOTS}
DATASET_PREFIX=${DATASET_PREFIX:-$ROOT/data/webdataset}

run_job "dinosaur_s$N_SLOTS" "$PARTITION_LONG" "${TIME:-48:00:00}" 1 24 -- \
  "export DATASET_PREFIX=$DATASET_PREFIX; \
   python -m ocl.cli.train +experiment=dinosaur/dinov3_small16_coco \
     models.conditioning.n_slots=$N_SLOTS trainer.max_steps=$MAX_STEPS \
     hydra.run.dir='$OUT' hydra.job.chdir=false"

echo "→ $OUT  (checkpoints under $OUT/checkpoints; copy the one you want to checkpoints/dinosaur_dinov3_vits16_coco.ckpt)"
