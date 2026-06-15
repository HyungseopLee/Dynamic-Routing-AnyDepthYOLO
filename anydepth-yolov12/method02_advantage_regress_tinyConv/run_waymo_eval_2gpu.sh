#!/bin/bash
# Waymo video eval, 2-way sharded across both GPUs (cuda:0, cuda:1) to halve wall
# time, then merge raw shards -> video_curve.json. Baselines + 5-seed policy + PI.
set -uo pipefail
source ~/anaconda3/bin/activate yolov12 2>/dev/null
cd /home/hslee/context-anydepth-det/anydepth-yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

O=method02_advantage_regress_tinyConv/outputs/waymo
W=finetuned_waymo/best.pt
POL=$(for s in 0 1 2 3 4; do echo -n "seed$s=$O/policy_$s.pt,"; done | sed 's/,$//')
mkdir -p "$O/eval"

run_shard () {
  # Pin each shard to ONE physical GPU via CUDA_VISIBLE_DEVICES so the process
  # only ever sees cuda:0 -> no cross-device (cuda:0 vs cuda:1) tensor mismatch.
  local SID=$1 GPU=$2
  CUDA_VISIBLE_DEVICES=$GPU python -u method02_advantage_regress_tinyConv/eval_video_waymo.py \
    --weight "$W" --policies "$POL" \
    --waymo_root /media/data/waymo_yolo/val \
    --grid 2 --imgsz 1280 1920 --conf 0.001 \
    --val_cache "$O/cache_val.pt" \
    --budgets 10,20,30,40,50,60,70,80,90 --pi \
    --num_shards 2 --shard_id "$SID" --device cuda:0 \
    --raw_out "$O/eval/shard_${SID}.pt" \
    > "$O/eval/video_shard_${SID}.log" 2>&1
}

echo "[$(date +%T)] launch 2 shards"
run_shard 0 0 &
P0=$!
run_shard 1 1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "[$(date +%T)] shards done (rc0=$R0 rc1=$R1)"

if [ $R0 -eq 0 ] && [ $R1 -eq 0 ]; then
  python method02_advantage_regress_tinyConv/merge_video_shards.py \
    --shards "$O/eval/shard_0.pt" "$O/eval/shard_1.pt" \
    --out "$O/eval/video_curve.json" 2>&1 | tee "$O/eval/merge.log"
  echo "[$(date +%T)] merged -> $O/eval/video_curve.json"
else
  echo "[$(date +%T)] a shard failed; not merging"
fi
