#!/bin/bash
# Robust, IDEMPOTENT Waymo video eval for the feat=BOTH router. Same 4-shard /
# 2-wave structure as run_waymo_eval_robust.sh, but uses the BOTH policies and the
# BOTH val cache (pred grids present, tapped online). Run in a tmux you own.
#
#   tmux new -s waymo_both
#   bash method02_advantage_regress_tinyConv/run_waymo_eval_both_robust.sh
set -uo pipefail
source ~/anaconda3/bin/activate yolov12 2>/dev/null
cd /home/hslee/context-anydepth-det/anydepth-yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

O=method02_advantage_regress_tinyConv/outputs/waymo
W=finetuned_waymo/best.pt
POL=$(for s in 0 1 2 3 4; do echo -n "seed$s=$O/policy_both_$s.pt,"; done | sed 's/,$//')
mkdir -p "$O/eval_both"

shard () {  # gpu, shard_id(0..3)
  local GPU=$1 SID=$2 RAW="$O/eval_both/shard_n${2}.pt"
  if [ -f "$RAW" ]; then echo "[$(date +%T)] shard $SID already done ($RAW), skip"; return 0; fi
  echo "[$(date +%T)] shard $SID on GPU$GPU -> $RAW"
  CUDA_VISIBLE_DEVICES=$GPU python -u method02_advantage_regress_tinyConv/eval_video_waymo.py \
    --weight "$W" --policies "$POL" --waymo_root /media/data/waymo_yolo/val \
    --grid 2 --imgsz 1280 1920 --conf 0.001 --val_cache "$O/cache_val_both.pt" \
    --budgets 10,20,30,40,50,60,70,80,90 --pi \
    --num_shards 4 --shard_id "$SID" --device cuda:0 \
    --raw_out "$RAW" > "$O/eval_both/video_shard_n${SID}.log" 2>&1
}

shard 0 0 & A=$!
shard 1 1 & B=$!
wait $A; wait $B
shard 0 2 & C=$!
shard 1 3 & D=$!
wait $C; wait $D

RAWS=("$O"/eval_both/shard_n0.pt "$O"/eval_both/shard_n1.pt "$O"/eval_both/shard_n2.pt "$O"/eval_both/shard_n3.pt)
miss=0; for r in "${RAWS[@]}"; do [ -f "$r" ] || { echo "MISSING $r"; miss=1; }; done
if [ $miss -eq 0 ]; then
  python method02_advantage_regress_tinyConv/merge_video_shards.py \
    --shards "${RAWS[@]}" --out "$O/eval_both/video_curve.json" 2>&1 | tee "$O/eval_both/merge.log"
  echo "[$(date +%T)] MERGED -> $O/eval_both/video_curve.json"
else
  echo "[$(date +%T)] some shards missing; re-run this script to finish them."
fi
