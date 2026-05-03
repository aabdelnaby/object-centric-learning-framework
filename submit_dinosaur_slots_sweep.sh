#!/bin/bash
# Sweep DINOSAUR (DINOv3 backbone) over different slot counts.
#
# Slot counts: 3, 11, 20, 30, 50, 100  (6 jobs → array index 0-5)
#
# Usage:
#   sbatch submit_dinosaur_slots_sweep.sh
#
# Outputs land in:
#   outputs/dinosaur_slots_sweep/slots_<N>/

#SBATCH --job-name=dino_slots_sweep
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_jobs_logs/dinosaur_slots_sweep/slurm-%A-%a.out
#SBATCH --error=slurm_jobs_logs/dinosaur_slots_sweep/slurm-%A-%a.err

source ~/.bashrc
conda activate oclf_env

cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework

# One slot count per array task
SLOT_COUNTS=(3 11 20 30 50 100)
N_SLOTS=${SLOT_COUNTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Job array task $SLURM_ARRAY_TASK_ID  |  n_slots=$N_SLOTS ==="

mkdir -p slurm_jobs_logs/dinosaur_slots_sweep

export DATASET_PREFIX=scripts/datasets/outputs

poetry run ocl_train \
    +experiment=projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto_dinov3_nohier \
    models.conditioning.n_slots=${N_SLOTS} \
    hydra.run.dir="outputs/dinosaur_slots_sweep/slots_${N_SLOTS}/${SLURM_ARRAY_JOB_ID}" \
    hydra.job.chdir=false
