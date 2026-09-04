#!/bin/bash
# Decoupled eval for submit_ood_sports_seed_range.sh. Same MODEL + SEED_OFFSET mapping.
#   sbatch --export=ALL,MODEL=proj,SEED_OFFSET=5 --array=0-4 \
#          --dependency=afterany:<train_arrayid> submit_ood_sports_seed_range_eval.sh
#SBATCH --job-name=sportSeedXEval
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%A_%a-ood_sports_seedX_eval.out
#SBATCH --error=slurm-%A_%a-ood_sports_seedX_eval.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

MODEL=${MODEL:?set MODEL via --export}
SEED_OFFSET=${SEED_OFFSET:-0}
SEED=$(( SLURM_ARRAY_TASK_ID + SEED_OFFSET ))
CKPT_DIR=runs/ood_sports_seeds/${MODEL}/seed${SEED}
OOD_CSV=FG-datset/ood_splits/paco_questions_sports_only.csv
DINO_CACHE=FG-datset/dino_feat_cache_combined_square.pt

BEST=$(ls $CKPT_DIR/patches/best_model.pt $CKPT_DIR/slots_*/best_model.pt 2>/dev/null | head -1)
echo "=== eval | model=$MODEL seed=$SEED | ckpt=$BEST ==="
if [ -z "$BEST" ]; then echo "NO CHECKPOINT for $MODEL/seed$SEED"; exit 1; fi
python eval_ood.py --checkpoint "$BEST" --csv_path "$OOD_CSV" --split val --dino_cache "$DINO_CACHE"
