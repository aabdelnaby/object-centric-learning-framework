#!/bin/bash
# Decoupled eval for the sports-only multi-seed sweep. Same array layout as
# submit_ood_sports_seeds.sh: task -> (model, seed). Evals that run's best_model.pt
# on the held-out sports set. Submit with a dependency on the training array:
#   sbatch --dependency=afterany:<train_arrayid> submit_ood_sports_seeds_eval.sh
#SBATCH --job-name=sportSeedEval
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --array=0-14
#SBATCH --output=slurm-%A_%a-ood_sports_seeds_eval.out
#SBATCH --error=slurm-%A_%a-ood_sports_seeds_eval.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

MODELS=(raw proj rtr)
MIDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))
MODEL=${MODELS[$MIDX]}
CKPT_DIR=runs/ood_sports_seeds/${MODEL}/seed${SEED}
OOD_CSV=FG-datset/ood_splits/paco_questions_sports_only.csv
DINO_CACHE=FG-datset/dino_feat_cache_combined_square.pt

BEST=$(ls $CKPT_DIR/patches/best_model.pt $CKPT_DIR/slots_*/best_model.pt 2>/dev/null | head -1)
echo "=== eval | model=$MODEL seed=$SEED | ckpt=$BEST ==="
if [ -z "$BEST" ]; then echo "NO CHECKPOINT for $MODEL/seed$SEED"; exit 1; fi

python eval_ood.py \
    --checkpoint "$BEST" \
    --csv_path   "$OOD_CSV" \
    --split      val \
    --dino_cache "$DINO_CACHE"
