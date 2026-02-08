#!/usr/bin/env bash

# Usage: ./submit_eval_and_visualize.sh <checkpoint_path> <train_config_path>
#
# Submits an eval job with the given checkpoint and config, waits for it to finish,
# moves generated *.npy files from ./outputs/val to ./outputs/val/<relevant_name>,
# and then runs visualize.py on that folder, saving PNGs to ./visualization_results/<relevant_name>.
#
# You can override some defaults via environment variables:
#   SLURM_PARTITION       (default: gpu_a100_short)
#   SLURM_NODES           (default: 1)
#   SLURM_TASKS_PER_NODE  (default: 1)
#   SLURM_CPUS            (default: 24)
#   SLURM_GPUS            (default: 4)
#   SLURM_TIME_MIN        (default: 30)
#   CONDA_ENV             (default: oclf_env)
#   DATASET_PREFIX        (default: scripts/datasets/outputs)
#   EVALUATION_OVERRIDE   (default: projects/bridging/outputs_movi_e.yaml)
#   WAIT_POLL_SEC         (default: 30)

SLURM_PARTITION=${SLURM_PARTITION:-gpu_a100_short}
SLURM_NODES=${SLURM_NODES:-1}
SLURM_TASKS_PER_NODE=${SLURM_TASKS_PER_NODE:-1}
SLURM_CPUS=${SLURM_CPUS:-24}
SLURM_GPUS=${SLURM_GPUS:-4}
SLURM_TIME_MIN=${SLURM_TIME_MIN:-30}
CONDA_ENV=${CONDA_ENV:-oclf_env}
DATASET_PREFIX=${DATASET_PREFIX:-scripts/datasets/outputs}
EVALUATION_OVERRIDE=${EVALUATION_OVERRIDE:-projects/bridging/outputs_coco_ccrop_slots_dinov3.yaml}
WAIT_POLL_SEC=${WAIT_POLL_SEC:-5}

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint_path> <train_config_path>" >&2
  exit 1
fi

CKPT_PATH=$1
TRAIN_CONFIG_PATH=$2

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi
if [[ ! -e "$TRAIN_CONFIG_PATH" ]]; then
  echo "Train config path not found: $TRAIN_CONFIG_PATH" >&2
  exit 1
fi

# Derive relevant name from checkpoint filename (without extension)
CKPT_BASE=$(basename "$CKPT_PATH")
RELEVANT_NAME=${CKPT_BASE%.ckpt}

LOG_DIR="slurm_jobs_logs/eval/${RELEVANT_NAME}"
mkdir -p "$LOG_DIR"

SBATCH_SCRIPT="$LOG_DIR/sbatch_eval_${RELEVANT_NAME}.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval-${RELEVANT_NAME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks-per-node=${SLURM_TASKS_PER_NODE}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --time=${SLURM_TIME_MIN}
#SBATCH --output=${LOG_DIR}/slurm-%j.out
#SBATCH --error=${LOG_DIR}/slurm-%j.err

source ~/.bashrc
conda activate ${CONDA_ENV}

export DATASET_PREFIX=${DATASET_PREFIX}

poetry run ocl_eval \
  +train_config_path=$(readlink -f "$TRAIN_CONFIG_PATH") \
  +evaluation=${EVALUATION_OVERRIDE} \
  +checkpoint_path=$(readlink -f "$CKPT_PATH") \
  +output_dir=.
EOF

chmod +x "$SBATCH_SCRIPT"

# Submit job and capture job ID
SUBMIT_OUTPUT=$(sbatch "$SBATCH_SCRIPT")
echo "$SUBMIT_OUTPUT"
JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')
if [[ -z "${JOB_ID:-}" ]]; then
  echo "Failed to parse job ID from sbatch output." >&2
  exit 1
fi

echo "Submitted job ${JOB_ID}. Waiting for completion..."

# Wait for job to complete (requires squeue)
if ! command -v squeue >/dev/null 2>&1; then
  echo "Warning: squeue not found. Skipping wait; continuing immediately." >&2
else
  while squeue -j "$JOB_ID" -h >/dev/null 2>&1; do
    sleep "$WAIT_POLL_SEC"
  done
fi

echo "Job ${JOB_ID} finished. Organizing outputs..."

# Move *.npy from ./outputs/val to ./outputs/val/<relevant_name>
VAL_DIR="./outputs/val"
DEST_DIR="${VAL_DIR}/${RELEVANT_NAME}"
mkdir -p "$DEST_DIR"

shopt -s nullglob
npys=("${VAL_DIR}"/*.npy)
if (( ${#npys[@]} == 0 )); then
  echo "No .npy files found in ${VAL_DIR}. Check logs: ${LOG_DIR}" >&2
else
  mv "${VAL_DIR}"/*.npy "$DEST_DIR"/
  echo "Moved ${#npys[@]} files to ${DEST_DIR}"
fi
shopt -u nullglob

echo "Running visualization..."
VIS_SAVE_DIR="./visualization_results/${RELEVANT_NAME}"
mkdir -p "$VIS_SAVE_DIR"

poetry run python visualize.py \
  --output_dir "$DEST_DIR" \
  --save_dir "$VIS_SAVE_DIR"

echo "Done. Visualizations saved to: $VIS_SAVE_DIR"
