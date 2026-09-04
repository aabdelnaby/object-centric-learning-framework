#!/bin/bash
# Paper-faithful VQA classifier on ADE20K part-COLOR questions, with recursive
# (zoom-in) inference — the ADE20K analogue of submit_superclevr3d_vqa_paper_t5_simple.sh.
#
# Data:
#   FG-datset/ade20k/parts_color_vqa.csv
#     7,240 images, 21,603 questions (19,189 train / 2,414 val),
#     1,117 unique questions, 21 train answer classes (colors), all attribute=color, depth=2.
#     Questions are part-level: "What is the color of the <part> of the <object>?"
#     val majority-class ("dark brown") baseline = 15.0% ; chance = 1/21 = 4.8%.
#   image_name is relative to the ADE20K_2021_17_01/images root and looks like
#   "ADE/training/<category>/<scene>/ADE_train_*.jpg" (wired in train.py's
#   DATASET_CONFIGS["ade20k"]).
#
# Model = the same paper-faithful classifier as the Super-CLEVR runs:
#   upstream  : frozen DINOSAUR slots, COCO-trained plain-DINO ViT-S/16 @224
#               (384-dim, object_dim 256), ckpt checkpoints/epoch_67-step_500000_coco.ckpt,
#               cfg projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto.
#               COCO is natural-image / in-domain for ADE20K scenes; RandomConditioning
#               means the 7-slot checkpoint runs at any --n_slots (7 sampled here).
#   text      : T5-base encoder (768-d), frozen
#   downstream: --pooler vqa_paper (d_model=128, ff=128, 64 heads, MLP head)
#   recursive : rank slots via the transformer pooler's CLS->slot attention, refine
#               the top --recursive_parents slot(s) into --recursive_children finer
#               child slots, then re-classify on parents+children. Matches the
#               part-of-object structure of the ADE20K questions.
#   training  : Adam, constant lr 1e-4, batch 128, cross-entropy (paper App. A.3).
#
# SUCCESS CRITERION: val acc >> 15.0% (majority baseline). If it stays near
# ~15%, the model isn't using the image (wiring/feature problem).
#
# Features: --text_in_memory encodes the 1,117 unique questions with T5 once at
# run start; --dino_in_memory runs the frozen COCO ViT over the 7,240 unique
# images once at run start. Both stay in RAM (shared by train+val); nothing is
# written to disk, so no feature cache and no /pfs quota / partial-write risk.
#
# Submit:
#   sbatch submit_ade20k_vqa_paper_t5_simple.sh
# This 30-min short slot makes meaningful progress (best/last checkpoint saved
# every epoch). For full convergence, switch to a longer partition and bump
# --max_steps; the run is resumable by adding --resume with a stable --checkpoint_dir.

#SBATCH --job-name=adevqa
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j-ade20k_vqa_part_color.out
#SBATCH --error=slurm-%j-ade20k_vqa_part_color.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# T-n downstream depth; TF-2 is fast and sufficient.
POOLER_LAYERS=2

CSV=FG-datset/paco_questions.csv
# COCO-trained plain-DINO ViT-S/16 @224 (384-dim) backbone — natural-image / in-domain.
DINO_CKPT=checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt
DINO_CFG=projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3
# Per-job checkpoint dir (SLURM_JOB_ID; falls back to "local" outside SLURM).
CKPT_DIR=runs/ade20k_vqa_coco_dino_T${POOLER_LAYERS}_part_color/single_pass/11_slots/${SLURM_JOB_ID:-local}

# ── Train + validate (per-epoch val accuracy is the signal) ──────────────────
# Text (T5) and COCO plain-DINO ViT-S/16 @224 image features are both computed
# in RAM at startup (--text_in_memory / --dino_in_memory); nothing is read from disk.
python train.py \
    --dataset          ade20k \
    --patch_control \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           vqa_paper \
    --rank_method      attribution \
    --pooler_layers    "$POOLER_LAYERS" \
    --n_slots          9 \
    --dinosaur_cfg     "$DINO_CFG" \
    --dinosaur_ckpt    "$DINO_CKPT" \
    --img_size         224 \
    --resize_mode      square \
    --feat_cache \
    --text_in_memory \
    --dino_cache     FG-datset/dino_feat_cache_combined_square.pt \
    --batch_size       128 \
    --lr               1e-4 \
    --warmup_steps     10000 \
    --max_steps        60000 \
    --patience         1000000 \
    --checkpoint_every 25 \
    --viz_n_samples 4 \
    --num_heads        64 \
    --skip_per_query_eval \
    --pooler_dropout 0.0 \
    --checkpoint_dir   "$CKPT_DIR"
