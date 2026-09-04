#!/bin/bash
# Evaluate the trained HierRouter best_model.pt on PACO val:
#   * top-1 / top-2 / top-3 accuracy for the MARGINAL P(a) and the BEST-PATH P(a|c_{j*k*})
#   * ~50 routing visualisations
# Quick (val=~2.7k rows, no training), 30 min is plenty.

#SBATCH --job-name=hrEval
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-eval_hier_router.out
#SBATCH --error=slurm-%j-eval_hier_router.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

CKPT=runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt

python eval_hier_router.py \
    --checkpoint "$CKPT" \
    --split      val \
    --batch_size 128 \
    --n_viz      50 \
    --device     cuda
