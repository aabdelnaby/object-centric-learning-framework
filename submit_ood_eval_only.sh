#!/bin/bash
# Decoupled OOD eval for the leave-one-scene-out sweep.
# The training jobs always hit their walltime (max_steps=500k never completes), so the
# in-job eval step never runs — but best_model.pt is saved continuously. This array evals
# each scene's saved best_model.pt on its held-out <scene>_only.csv.
#
# Pick the model family with --export:  MODEL=baseline | baseline_projected | router
#   sbatch --export=ALL,MODEL=baseline submit_ood_eval_only.sh
#   sbatch --export=ALL,MODEL=baseline_projected --dependency=afterany:<proj_train_arrayid> submit_ood_eval_only.sh
#   sbatch --export=ALL,MODEL=router  --dependency=afterany:<router_train_arrayid> submit_ood_eval_only.sh
#SBATCH --job-name=oodEval
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --array=0-12
#SBATCH --output=slurm-%A_%a-ood_eval_%x.out
#SBATCH --error=slurm-%A_%a-ood_eval_%x.err

source ~/.bashrc
conda activate internvl_env 2>/dev/null; conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

MODEL=${MODEL:-baseline}
SCENES=(kitchen living_room restaurant street office park_nature bedroom \
        bathroom dining_room store sports vehicle beach_water)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
DINO_CACHE=FG-datset/dino_feat_cache_combined_square.pt
OOD_CSV=FG-datset/ood_splits/paco_questions_${SCENE}_only.csv

if [ "$MODEL" = "router" ]; then
    BEST=$(ls runs/ood_sweep_hier_router/$SCENE/*/slots_*/best_model.pt 2>/dev/null | head -1)
elif [ "$MODEL" = "baseline_projected" ]; then
    BEST=$(ls runs/ood_sweep_baseline_projected/$SCENE/*/patches/best_model.pt 2>/dev/null | head -1)
else
    BEST=$(ls runs/ood_sweep_baseline/$SCENE/*/patches/best_model.pt 2>/dev/null | head -1)
fi

echo "=== OOD eval | model=$MODEL | scene=$SCENE | ckpt=$BEST ==="
if [ -z "$BEST" ]; then echo "NO CHECKPOINT for $MODEL/$SCENE"; exit 1; fi

python eval_ood.py \
    --checkpoint "$BEST" \
    --csv_path   "$OOD_CSV" \
    --split      val \
    --dino_cache "$DINO_CACHE"
