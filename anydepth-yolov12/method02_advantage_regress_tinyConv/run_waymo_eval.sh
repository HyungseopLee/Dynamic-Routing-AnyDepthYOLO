#!/bin/bash
# Streaming depth-routing video eval on Waymo val (our protocol, Waymo labels).
# Mirrors the BDD eval: per-segment causal routing for all 5-seed policies +
# PI latency-budget control. Single-GPU full run (202 segments) -> curve json.
set -uo pipefail
source ~/anaconda3/bin/activate yolov12 2>/dev/null
cd /home/hslee/context-anydepth-det/anydepth-yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

O=method02_advantage_regress_tinyConv/outputs/waymo
W=finetuned_waymo/best.pt
POL=$(for s in 0 1 2 3 4; do echo -n "seed$s=$O/policy_$s.pt,"; done | sed 's/,$//')
mkdir -p "$O/eval"
LOG="$O/eval/video_eval.log"

echo "[$(date +%T)] waymo video eval  weight=$W" | tee "$LOG"

python -u method02_advantage_regress_tinyConv/eval_video_waymo.py \
    --weight "$W" --policies "$POL" \
    --waymo_root /media/data/waymo_yolo/val \
    --grid 2 --imgsz 1280 1920 --conf 0.001 \
    --val_cache "$O/cache_val.pt" \
    --budgets 10,20,30,40,50,60,70,80,90 \
    --pi \
    --out "$O/eval/video_curve.json" \
    2>&1 | tee -a "$LOG"

echo "[$(date +%T)] waymo video eval done -> $O/eval/video_curve.json" | tee -a "$LOG"
