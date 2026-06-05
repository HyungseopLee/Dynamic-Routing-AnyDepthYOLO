#!/bin/bash
# SLURM job array: train multiple policy configs in parallel, 1 GPU per task,
# up to 4 concurrent (%4). Each array index runs ONE config from CONFIGS below.
#
# Policy training reads precomputed caches (tiny tinyConv+MLP), so 1 GPU/run is
# plenty; the array just runs MANY runs (seeds/ablations) at once.
#
# Usage:
#   cd anydepth-yolov12
#   sbatch method02_advantage_regress_tinyConv/train_policy_sbatch.sh
# Adjust --array range to match the number of CONFIGS entries (0-indexed).

#SBATCH --job-name=policy_bdd
#SBATCH --output=method02_advantage_regress_tinyConv/outputs/bdd100k/logs/slurm_%A_%a.out
#SBATCH --error=method02_advantage_regress_tinyConv/outputs/bdd100k/logs/slurm_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # 1 GPU per array task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4%4           # 5 runs (seeds 0..4), max 4 concurrent

set -euo pipefail
source ~/miniconda3/bin/activate yolov12 2>/dev/null || source ~/anaconda3/bin/activate yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# ---- config list: one line per array task -----------------------------------
# backbone-only (feat=input) tinyConv on grid-2 cache. Edit/extend freely;
# keep --array range in sync with the number of entries.
CONFIGS=(
  "--seed 0 --tag seed0"
  "--seed 1 --tag seed1"
  "--seed 2 --tag seed2"
  "--seed 3 --tag seed3"
  "--seed 4 --tag seed4"
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[array $SLURM_ARRAY_TASK_ID] config: $CFG  (GPU: ${CUDA_VISIBLE_DEVICES:-?})"

# shared args: backbone-only policy on the bdd100k cache (1 GPU = cuda:0 in task)
COMMON="--dataset bdd100k --feat input --norm batch \
        --epochs 50 --select val_corr --mode regress --regress_loss mse \
        --device cuda:0"

# human-readable per-run log alongside the SLURM .out/.err
LOGDIR=method02_advantage_regress_tinyConv/outputs/bdd100k/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/policy_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.log"

python method02_advantage_regress_tinyConv/train_policy.py $COMMON $CFG \
    --out method02_advantage_regress_tinyConv/outputs/bdd100k/policy_${SLURM_ARRAY_TASK_ID}.pt \
    2>&1 | tee "$LOG"
