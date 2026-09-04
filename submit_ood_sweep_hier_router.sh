#!/bin/bash
# Leave-one-scene-out OOD sweep — hier_router (readout) model under test.
# SLURM job array: task i holds out SCENES[i], trains on paco_questions_no_<scene>.csv
# (train+val, that scene removed), then evals best_model.pt on <scene>_only.csv.
# Partner sweep: submit_ood_sweep_baseline.sh (identical CSVs/cache, flat control).
#SBATCH --job-name=oodSweepRtr
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-12
#SBATCH --output=slurm-%A_%a-ood_sweep_hier_router.out
#SBATCH --error=slurm-%A_%a-ood_sweep_hier_router.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

SCENES=(kitchen living_room restaurant street office park_nature bedroom \
        bathroom dining_room store sports vehicle beach_water)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
echo "=== Held-out scene: $SCENE (array task $SLURM_ARRAY_TASK_ID) ==="

POOLER_LAYERS=5
N_SLOTS=7
TRAIN_CSV=FG-datset/ood_splits/paco_questions_no_${SCENE}.csv
OOD_CSV=FG-datset/ood_splits/paco_questions_${SCENE}_only.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
DINO_CACHE=FG-datset/dino_feat_cache_combined_square.pt
CKPT_DIR=runs/ood_sweep_hier_router/${SCENE}/${SLURM_ARRAY_JOB_ID}

# ── Train + in-domain val ($SCENE NOT in this CSV) → best_model.pt ────────────
python train.py \
    --dataset          ade20k \
    --csv_path         "$TRAIN_CSV" \
    --text_encoder     t5 \
    --pooler           hier_router \
    --rank_method      attribution \
    --pooler_layers    "$POOLER_LAYERS" \
    --n_slots          "$N_SLOTS" \
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
    --viz_n_samples 4 \
    --num_heads        64 \
    --skip_per_query_eval \
    --recursive_children 4 \
    --recursive_spread 0.0 \
    --pooler_dropout 0.0 \
    --router_temp 0.95 \
    --child_scorer mlp \
    --weight_decay 0.02 \
    --checkpoint_dir   "$CKPT_DIR"

# ── OOD eval: best checkpoint on the held-out scene ──────────────────────────
BEST=$CKPT_DIR/slots_${N_SLOTS}/best_model.pt
echo "=== OOD eval ($SCENE): $BEST ==="
python eval_ood.py \
    --checkpoint "$BEST" \
    --csv_path   "$OOD_CSV" \
    --split      val \
    --dino_cache "$DINO_CACHE"
