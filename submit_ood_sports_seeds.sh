#!/bin/bash
# Sports-only multi-seed OOD sweep: 3 models x 5 seeds = 15 training jobs.
# Each task holds out 'sports' (trains on paco_questions_no_sports.csv) with a fixed --seed,
# so we can report mean +/- std OOD accuracy per model and test whether the router's sports
# edge is real or within seed noise. Eval is decoupled (submit_ood_sports_seeds_eval.sh).
#
# array index -> (model, seed):  model = idx/5 (0 raw, 1 proj, 2 rtr),  seed = idx%5 (0..4)
#SBATCH --job-name=sportSeed
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-14
#SBATCH --output=slurm-%A_%a-ood_sports_seeds.out
#SBATCH --error=slurm-%A_%a-ood_sports_seeds.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

MODELS=(raw proj rtr)
MIDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))
MODEL=${MODELS[$MIDX]}
echo "=== sports holdout | model=$MODEL | seed=$SEED (task $SLURM_ARRAY_TASK_ID) ==="

TRAIN_CSV=FG-datset/ood_splits/paco_questions_no_sports.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
DINO_CACHE=FG-datset/dino_feat_cache_combined_square.pt
CKPT_DIR=runs/ood_sports_seeds/${MODEL}/seed${SEED}

case $MODEL in
  raw)  POOLER_ARGS="--pooler patch_qdot --patch_control --num_heads 8" ;;
  proj) POOLER_ARGS="--pooler patch_qdot --patch_control --patch_qdot_project_patches --num_heads 8" ;;
  rtr)  POOLER_ARGS="--pooler hier_router --rank_method attribution --pooler_layers 5 \
                     --n_slots 7 --num_heads 64 --recursive_children 4 --recursive_spread 0.0 \
                     --pooler_dropout 0.0 --router_temp 0.95 --child_scorer mlp --viz_n_samples 4" ;;
esac

python train.py \
    --dataset          ade20k \
    --csv_path         "$TRAIN_CSV" \
    --text_encoder     t5 \
    $POOLER_ARGS \
    --dinosaur_cfg     "$DINO_CFG" \
    --dinosaur_ckpt    "$DINO_CKPT" \
    --img_size         224 \
    --resize_mode      square \
    --feat_cache \
    --dino_cache       "$DINO_CACHE" \
    --text_in_memory \
    --batch_size       128 \
    --lr               2e-4 \
    --warmup_steps     10000 \
    --max_steps        500000 \
    --patience         1000000 \
    --checkpoint_every 25 \
    --skip_per_query_eval \
    --weight_decay     0.02 \
    --seed             "$SEED" \
    --checkpoint_dir   "$CKPT_DIR"
