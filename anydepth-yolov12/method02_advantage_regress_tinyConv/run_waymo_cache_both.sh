#!/bin/bash
# Build feat=BOTH router caches for Waymo (backbone input + neck pred grids),
# mirroring the BDD `_both` recipe: grid 2x2, add pred_base/pred_super (640ch).
# Writes to cache_{split}_both.pt so the existing feat=input caches are untouched.
# Heavy: ~40k val + ~158k train frames, BASE+SUPER forward each (saved once/split).
set -uo pipefail
source ~/anaconda3/bin/activate yolov12 2>/dev/null
cd /home/hslee/context-anydepth-det/anydepth-yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

W=finetuned_waymo/best.pt
DATA=ultralytics/cfg/datasets/waymo.yaml
O=method02_advantage_regress_tinyConv/outputs/waymo
mkdir -p "$O/logs"

for SPLIT in val train; do
  OUT="$O/cache_${SPLIT}_both.pt"
  if [ -f "$OUT" ]; then echo "[$(date +%T)] $OUT exists, skip"; continue; fi
  LOG="$O/logs/cache_${SPLIT}_both.log"
  echo "[$(date +%T)] build BOTH cache $SPLIT -> $OUT" | tee "$LOG"
  python method02_advantage_regress_tinyConv/build_cache.py \
    --weight "$W" --data "$DATA" --split "$SPLIT" \
    --imgsz 1280 1920 --grid 2 --feat both \
    --batch 16 --device cuda:0 --dataset waymo \
    --out "$OUT" \
    2>&1 | tee -a "$LOG"
done
echo "[$(date +%T)] both caches done"
