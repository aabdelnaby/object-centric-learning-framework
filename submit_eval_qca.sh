#!/bin/bash
# Evaluate both QCA baseline checkpoints on PACO val:
#   * top-1 / top-2 / top-3 accuracy
#   * ~50 attention-map figures per checkpoint (the area each model used to answer)
# Both runs sequential in one job; gpu_a100_short 30 min is plenty.

#SBATCH --job-name=qcaEval
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-eval_qca.out
#SBATCH --error=slurm-%j-eval_qca.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

PARENTSLOT_CKPT=runs/ade20k_qca_parentslot/4995862/slots_9/best_model.pt
PATCH_CKPT=runs/ade20k_qca_patch/4995863/patches/best_model.pt

echo "==================== ParentSlot-QCA ===================="
python eval_qca.py \
    --checkpoint "$PARENTSLOT_CKPT" \
    --split      val \
    --batch_size 128 \
    --n_viz      50 \
    --device     cuda

echo ""
echo "==================== Patch-QCA ========================="
python eval_qca.py \
    --checkpoint "$PATCH_CKPT" \
    --split      val \
    --batch_size 128 \
    --n_viz      50 \
    --device     cuda
