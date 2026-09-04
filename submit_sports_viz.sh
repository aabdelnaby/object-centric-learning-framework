#!/bin/bash
#SBATCH --job-name=sportsViz
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-sports_viz.out
#SBATCH --error=slurm-%j-sports_viz.err
source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

SPORTS_CSV=FG-datset/ood_splits/paco_questions_sports_only.csv
DINO_CACHE=FG-datset/dino_feat_cache_combined_square.pt
RTR_CKPT=runs/ood_sports_seeds/rtr/seed0/slots_7/best_model.pt
PROJ_CKPT=runs/ood_sports_seeds/proj/seed0/patches/best_model.pt

echo "########## ROUTER routing viz (sports holdout, ~100) ##########"
python eval_hier_router.py \
    --checkpoint "$RTR_CKPT" \
    --csv_path   "$SPORTS_CSV" \
    --split      val \
    --n_viz      100 \
    --dino_cache "$DINO_CACHE" \
    --out_dir    viz_sports/router || echo "ROUTER VIZ FAILED"

echo ""; echo "########## PROJECTED patch-attention viz (sports holdout, ~100) ##########"
python visualize_patch_qdot.py \
    --checkpoint "$PROJ_CKPT" \
    --csv_path   "$SPORTS_CSV" \
    --split      val \
    --n_samples  100 \
    --device     cuda \
    --out_dir    viz_sports/projected || echo "PROJECTED VIZ FAILED"

echo ""; echo "=== counts ==="
echo "router PNGs:    $(find viz_sports/router -name '*.png' 2>/dev/null | wc -l)"
echo "projected PNGs: $(find viz_sports/projected -name '*.png' 2>/dev/null | wc -l)"
