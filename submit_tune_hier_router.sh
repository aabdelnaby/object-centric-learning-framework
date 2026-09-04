#!/bin/bash
# Distributed Optuna search for the hier_router head on ADE20K (square-resize features).
#
# A SLURM ARRAY of workers all pull trials from one shared Optuna study (JournalStorage
# file on /pfs). Each worker shells out to train.py per trial with sampled HPs —
# n_slots (= parent-slot count P, since hier_router refines all slots), recursive_children
# (= child count K), lr, router_temp, child_scorer, router_entropy_weight, weight_decay —
# loads the precomputed square ViT/T5 caches, trains a short budget, and is scored by the
# best val acc in that run's metrics.csv (with median-pruning of weak trials).
#
# Prereq: run submit_precompute_ade20k_square.sh first (builds the caches).
#
#   sbatch submit_tune_hier_router.sh
#
# %4 = at most 4 array tasks run at once; 16 total. Resubmit to add more trials —
# they accumulate in the same study. If your partition grants more than ~30 min,
# raise --time AND TIMEOUT (and/or --n_trials) so each worker completes more trials.

#SBATCH --job-name=adetune
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-15%4
#SBATCH --output=slurm-%A_%a-ade20k_tune.out
#SBATCH --error=slurm-%A_%a-ade20k_tune.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# Self-stop ~25 min into the 30-min slot so the worker exits cleanly (an in-flight
# trial may still be cut by SLURM; completed trials are already saved in the study).
TIMEOUT=1500

python tune_hier_router.py \
    --study_name  hier_router_square \
    --storage     runs/optuna/hier_router_square/journal.log \
    --out_root    runs/optuna/hier_router_square \
    --dino_cache  FG-datset/ade20k/dino_feat_cache_square.pt \
    --text_cache  FG-datset/ade20k/text_feat_cache_t5_spans.pt \
    --resize_mode square \
    --max_epochs  50 \
    --patience    20 \
    --n_trials    8 \
    --timeout     "$TIMEOUT" \
    --seed        "${SLURM_ARRAY_TASK_ID:-0}"
