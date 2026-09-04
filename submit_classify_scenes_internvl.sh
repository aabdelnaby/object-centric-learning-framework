#!/bin/bash
# Scene classification of every unique paco_questions.csv image with InternVL3-14B
# (image + COCO captions -> one scene label). SLURM job array: each task is one
# shard; together the array covers all 22,380 unique images. Resumable
# (--skip-done), so re-submitting fills any shard that ran out of time.
#
# NUM_SHARDS must equal the array width (0..NUM_SHARDS-1). Sized so each shard
# fits the 30-min gpu_a100_short limit incl. ~2-3 min model load.
#SBATCH --job-name=sceneVLM
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-11
#SBATCH --output=slurm-%A_%a-scene_vlm.out
#SBATCH --error=slurm-%A_%a-scene_vlm.err

source ~/.bashrc
conda activate internvl_env
cd /pfs/data6/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

NUM_SHARDS=12

python -u classify_scenes_internvl.py \
  --csv FG-datset/paco_questions.csv \
  --out-dir FG-datset/preds_scene \
  --shard ${SLURM_ARRAY_TASK_ID} \
  --num-shards ${NUM_SHARDS} \
  --skip-done
