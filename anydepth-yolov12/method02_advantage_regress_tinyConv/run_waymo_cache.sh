#!/bin/bash
# Build router feature caches for Waymo (FRONT, native 1280x1920), mirroring the
# BDD recipe: grid 2, backbone (input) features, from the finetuned detector.
# Heavy: ~40k val + ~158k train frames, BASE+SUPER forward each.
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
  LOG="$O/logs/cache_${SPLIT}.log"
  echo "[$(date +%T)] build cache $SPLIT -> $O/cache_${SPLIT}.pt" | tee "$LOG"
  python method02_advantage_regress_tinyConv/build_cache.py \
    --weight "$W" --data "$DATA" --split "$SPLIT" \
    --imgsz 1280 1920 --grid 2 --feat input \
    --batch 16 --device cuda:0 --dataset waymo \
    2>&1 | tee -a "$LOG"
done
echo "[$(date +%T)] caches done"
