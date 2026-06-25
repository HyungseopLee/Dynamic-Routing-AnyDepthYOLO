#!/usr/bin/env bash
# Waymo video eval for feat=both router. 4-shard / 2-wave parallel structure.
# Idempotent: already-completed shards are skipped.
#
# Usage (from repo root):
#   tmux new -s waymo_both
#   bash method_advantage_regress/eval/run_waymo_eval_both_robust.sh
set -e
cd "$(dirname "$0")/../.."

PKG=method_advantage_regress
O=$PKG/outputs/waymo
W=finetuning_AnyDepthYOLO/weights/waymo/best.pt
POL=$(for s in 0 1 2 3 4; do echo -n "seed$s=$O/policy_both_$s.pt,"; done | sed 's/,$//')
mkdir -p "$O/eval_both"

shard() {  # gpu, shard_id(0..3)
  local GPU=$1 SID=$2 RAW="$O/eval_both/shard_n${2}.pt"
  if [ -f "$RAW" ]; then echo "[$(date +%T)] shard $SID already done, skip"; return 0; fi
  echo "[$(date +%T)] shard $SID on GPU$GPU -> $RAW"
  CUDA_VISIBLE_DEVICES=$GPU python -u -m $PKG.eval_video --dataset waymo \
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

RAWS=("$O"/eval_both/shard_n{0,1,2,3}.pt)
miss=0; for r in "${RAWS[@]}"; do [ -f "$r" ] || { echo "MISSING $r"; miss=1; }; done
if [ $miss -eq 0 ]; then
  python -m $PKG.merge_video_shards \
    --shards "${RAWS[@]}" --out "$O/eval_both/video_curve.json" 2>&1 | tee "$O/eval_both/merge.log"
  echo "[$(date +%T)] MERGED -> $O/eval_both/video_curve.json"
else
  echo "[$(date +%T)] some shards missing; re-run to finish them."
fi
