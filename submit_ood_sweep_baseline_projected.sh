#!/bin/bash
# Leave-one-scene-out OOD sweep — BASELINE (Patch-QDot PROJECTED).
# Identical to submit_ood_sweep_baseline.sh EXCEPT --patch_qdot_project_patches (learned
# d_vit->d_slot projection before the query·patch dot-product). Third model alongside the
# raw baseline and the hier_router. SLURM array: task i holds out SCENES[i].
#SBATCH --job-name=oodSweepProj
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-12
#SBATCH --output=slurm-%A_%a-ood_sweep_baseline_projected.out
#SBATCH --error=slurm-%A_%a-ood_sweep_baseline_projected.err

source ~/.bashrc
conda activate oclf_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

SCENES=(kitchen living_room restaurant street office park_nature bedroom \
        bathroom dining_room store sports vehicle beach_water)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
echo "=== Held-out scene: $SCENE (array task $SLURM_ARRAY_TASK_ID) ==="

TRAIN_CSV=FG-datset/ood_splits/paco_questions_no_${SCENE}.csv
OOD_CSV=FG-datset/ood_splits/paco_questions_${SCENE}_only.csv
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
DINO_CACHE=FG-datset/dino_feat_cache_combined_square.pt
CKPT_DIR=runs/ood_sweep_baseline_projected/${SCENE}/${SLURM_ARRAY_JOB_ID}

# ── Train + in-domain val ($SCENE NOT in this CSV) → best_model.pt ────────────
python train.py \
    --dataset          ade20k \
    --csv_path         "$TRAIN_CSV" \
    --text_encoder     t5 \
    --pooler           patch_qdot \
    --patch_control \
    --patch_qdot_project_patches \
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
    --num_heads        8 \
    --skip_per_query_eval \
    --weight_decay 0.02 \
    --checkpoint_dir   "$CKPT_DIR"

# (in-job eval intentionally omitted — training hits walltime; eval is decoupled
#  via submit_ood_eval_only.sh with MODEL=baseline_projected.)
