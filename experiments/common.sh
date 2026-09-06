#!/bin/bash
# Shared settings + a SLURM launcher for the numbered experiment scripts. Source it:
#   source "$(dirname "$0")/common.sh"
#
# Environment variables you may override:
#   CONDA_ENV        conda environment with the pinned dependencies      (default oclf_env)
#   PARTITION_SHORT  A100 partition for <= 30 min jobs                    (default gpu_a100_short)
#   PARTITION_LONG   A100 partition for multi-hour jobs                    (default gpu_a100_il)
#   PARTITION_CPU    CPU-only partition for dataset construction           (default cpu)
#   RUNS_ROOT        where training runs are written                       (default <repo>/runs)
#   RESULTS          where evaluation JSONs / tables are written           (default <repo>/results)
#   CKPT_DIR         trained heads used by 07/08 (default checkpoints/thesis, see tools/)
#   LOCAL=1          run the command in the foreground instead of via sbatch
#   DRY=1            print the sbatch command instead of submitting
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_ENV=${CONDA_ENV:-oclf_env}
PARTITION_SHORT=${PARTITION_SHORT:-gpu_a100_short}
PARTITION_LONG=${PARTITION_LONG:-gpu_a100_il}
PARTITION_CPU=${PARTITION_CPU:-cpu}
RUNS_ROOT=${RUNS_ROOT:-$ROOT/runs}
RESULTS=${RESULTS:-$ROOT/results}
CKPT_DIR=${CKPT_DIR:-$ROOT/checkpoints/thesis}
LOGS=${LOGS:-$ROOT/slurm_logs}
export ROOT CONDA_ENV RUNS_ROOT RESULTS CKPT_DIR

# run_job NAME PARTITION TIME N_GPUS N_CPUS -- COMMAND...
# Submits COMMAND (run from the repo root inside CONDA_ENV) as one SLURM job and prints the
# job id. Extra sbatch options can be passed through SBATCH_EXTRA (e.g. "--array=0-7").
run_job() {
  local name=$1 partition=$2 time=$3 gpus=$4 cpus=$5
  shift 5
  [ "$1" = "--" ] && shift
  local cmd="$*"
  if [ "${LOCAL:-0}" = "1" ]; then
    (cd "$ROOT" && eval "$cmd")
    return $?
  fi
  mkdir -p "$LOGS"
  local gres=()
  [ "$gpus" -gt 0 ] && gres=(--gres="gpu:$gpus")
  local args=(--job-name="$name" --partition="$partition" --time="$time" --nodes=1 --ntasks=1
              --cpus-per-task="$cpus" "${gres[@]}" --output="$LOGS/%j-$name.out" --error="$LOGS/%j-$name.err"
              ${SBATCH_EXTRA:-})
  local wrap="source ~/.bashrc; conda activate $CONDA_ENV; cd $ROOT; export PYTHONUNBUFFERED=1; $cmd"
  if [ "${DRY:-0}" = "1" ]; then
    echo "sbatch ${args[*]} --wrap=\"$wrap\""
    return 0
  fi
  sbatch "${args[@]}" --wrap="$wrap"
}
