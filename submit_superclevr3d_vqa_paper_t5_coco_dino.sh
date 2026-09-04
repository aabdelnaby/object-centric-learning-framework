#!/bin/bash
# Recursive (zoom-in) VQA classifier on Super-CLEVR-3D simple-color part questions,
# using the COCO-trained plain-DINO DINOSAUR backbone (out-of-domain sanity check).
#
# Backbone for this script:
#   frozen DINOSAUR slots, plain DINO ViT-S/16 @224 (384-dim), object_dim 256,
#   ckpt checkpoints/epoch_67-step_500000_coco.ckpt
#   cfg  projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto
#   (RandomConditioning → runs at any --n_slots; this checkpoint was trained at 7,
#    so 11 slots are sampled from the same learned init at inference.)
#
# text       : T5-base encoder (768-d), frozen
# downstream : --pooler vqa_paper (d_model=128, ff=128, 64 heads, MLP head)
# recursive  : rank slots via the transformer pooler, refine the top --recursive_parents
#              slots into --recursive_children each, re-classify on parents+children.
# training   : Adam, constant lr 1e-4, batch 128, cross-entropy (paper App. A.3),
#              20% train subset (--train_frac 0.2) for a quick check.
#
# NOTE: COCO is natural-image / out-of-domain for synthetic CLEVR scenes, so the
# slots may be messier than an sc3d-native backbone — this is a comparison run.
#
# Submit:
#   sbatch submit_superclevr3d_vqa_paper_t5_coco_dino.sh
# Resumable via --resume.

#SBATCH --job-name=scvqa_coco
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-scvqa_coco_dino.out
#SBATCH --error=slurm-%j-scvqa_coco_dino.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# T-n downstream depth; TF-2 is fast and sufficient for a sanity check.
POOLER_LAYERS=5

CSV=FG-datset/superclevr3d/simple_color_vqa.csv
# COCO-trained plain-DINO ViT-S/16 @224 (384-dim) backbone.
DINO_CKPT=checkpoints/epoch_67-step_500000_coco.ckpt
DINO_CFG=projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto
# Per-job checkpoint dir (SLURM_JOB_ID; falls back to "local" outside SLURM).
CKPT_DIR=runs/recursive_superclevr_vqa_coco_dino_T${POOLER_LAYERS}/topk/${SLURM_JOB_ID:-local}

# ── Train + validate (per-epoch val accuracy is the sanity signal) ───────────
# Text (T5) and plain-DINO ViT-S/16 @224 image features are computed in RAM at
# startup (--text_in_memory / --dino_in_memory); nothing is read from disk.
python train.py \
    --dataset          superclevr3d \
    --csv_path         "$CSV" \
    --text_encoder     t5 \
    --pooler           vqa_paper \
    --pooler_layers    "$POOLER_LAYERS" \
    --n_slots          11 \
    --dinosaur_cfg     "$DINO_CFG" \
    --dinosaur_ckpt    "$DINO_CKPT" \
    --img_size         224 \
    --feat_cache \
    --text_in_memory \
    --dino_in_memory \
    --batch_size       128 \
    --optimizer        adam \
    --lr_schedule      constant \
    --lr               1e-4 \
    --warmup_steps     10000 \
    --max_steps        10000 \
    --patience         1000000 \
    --checkpoint_every 1 \
    --viz_n_samples 4 \
    --num_heads        64 \
    --skip_per_query_eval \
    --recursive_infer \
    --recursive_children 5 \
    --recursive_parents 3 \
    --recursive_spread 0.0 \
    --train_frac 0.2 \
    --checkpoint_dir   "$CKPT_DIR"
