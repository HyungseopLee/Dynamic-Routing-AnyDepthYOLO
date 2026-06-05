#!/bin/bash
# SLURM job array: BDD100K MOT video eval for the depth-routing policy.
# Shared server -> always go through the scheduler, never run the detector
# forwards directly on a login/compute node.
#
# 4-way sharding: each array task evaluates a round-robin 1/4 of the 200 videos
# on 1 GPU and dumps raw per-strategy matches to shard_<id>.pt. After all four
# finish, merge into the final curve:
#
#   sbatch method02_advantage_regress_tinyConv/eval_video_bdd_sbatch.sh
#   # once the array completes:
#   python method02_advantage_regress_tinyConv/merge_video_shards.py \
#       --shards method02_advantage_regress_tinyConv/outputs/bdd100k/eval/shard_*.pt \
#       --out    method02_advantage_regress_tinyConv/outputs/bdd100k/eval/video_curve.json
#
# Single-GPU full run instead (e.g. a personal server): just call the python with
# --num_shards 1 (default) and --out, no array needed.

#SBATCH --job-name=eval_bdd_video
#SBATCH --output=method02_advantage_regress_tinyConv/outputs/bdd100k/eval/slurm_%A_%a.out
#SBATCH --error=method02_advantage_regress_tinyConv/outputs/bdd100k/eval/slurm_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3%4            # 4 shards, all 4 concurrent (one GPU each)

set -euo pipefail
source ~/miniconda3/bin/activate yolov12 2>/dev/null || source ~/anaconda3/bin/activate yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

O=method02_advantage_regress_tinyConv/outputs/bdd100k
W=$(ls finetuned_bdd100k/*alpha0.2*mAP35.1*.pt)
POL=$(for s in 0 1 2 3 4; do echo -n "seed$s=$O/policy_$s.pt,"; done | sed 's/,$//')
mkdir -p "$O/eval"
NS=4
SID=${SLURM_ARRAY_TASK_ID:-0}
LOG="$O/eval/video_shard_${SID}.log"

echo "[$(date +%T)] shard $SID/$NS  weight=$W"

python method02_advantage_regress_tinyConv/eval_video_bdd.py \
    --weight "$W" --policies "$POL" \
    --grid 2 --imgsz 720 1280 --conf 0.001 \
    --val_cache "$O/cache_val.pt" --budgets 10,20,30,40,50,60,70,80,90 \
    --num_shards $NS --shard_id $SID \
    --raw_out "$O/eval/shard_${SID}.pt" \
    2>&1 | tee "$LOG"
