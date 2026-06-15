#!/bin/bash
# Finish the Waymo video eval after shard0 died. shard1 (num_shards=2, id=1) is
# still running and covers the ODD segments -> shard_1.pt. The EVEN half is split
# 4-way into two disjoint sub-shards (num_shards=4: id0=[0::4], id2=[2::4]) so it
# can run on both GPUs. A starts now on GPU0; B starts on GPU1 once shard1 frees
# it. Then merge shard_1 + shard_a + shard_b -> video_curve.json (disjoint, full
# 202-segment cover).
set -uo pipefail
source ~/anaconda3/bin/activate yolov12 2>/dev/null
cd /home/hslee/context-anydepth-det/anydepth-yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

O=method02_advantage_regress_tinyConv/outputs/waymo
W=finetuned_waymo/best.pt
POL=$(for s in 0 1 2 3 4; do echo -n "seed$s=$O/policy_$s.pt,"; done | sed 's/,$//')

run_subshard () {  # gpu, shard_id(of num_shards=4), raw_tag
  local GPU=$1 SID=$2 TAG=$3
  CUDA_VISIBLE_DEVICES=$GPU python -u method02_advantage_regress_tinyConv/eval_video_waymo.py \
    --weight "$W" --policies "$POL" --waymo_root /media/data/waymo_yolo/val \
    --grid 2 --imgsz 1280 1920 --conf 0.001 --val_cache "$O/cache_val.pt" \
    --budgets 10,20,30,40,50,60,70,80,90 --pi \
    --num_shards 4 --shard_id "$SID" --device cuda:0 \
    --raw_out "$O/eval/shard_${TAG}.pt" \
    > "$O/eval/video_shard_${TAG}.log" 2>&1
}

echo "[$(date +%T)] launch sub-shard A (num4 id0) on GPU0"
run_subshard 0 0 a &
PA=$!

# wait for the still-running num2/id1 shard to finish and free GPU1
echo "[$(date +%T)] waiting for running shard1 (shard_1.pt) before using GPU1"
until [ -f "$O/eval/shard_1.pt" ]; do
  if ! pgrep -f "eval_video_waymo.py.*shard_id 1\b" >/dev/null && [ ! -f "$O/eval/shard_1.pt" ]; then
    echo "[$(date +%T)] WARNING shard1 proc gone and no shard_1.pt"; break
  fi
  sleep 60
done

echo "[$(date +%T)] launch sub-shard B (num4 id2) on GPU1"
run_subshard 1 2 b &
PB=$!

wait $PA; RA=$?
wait $PB; RB=$?
echo "[$(date +%T)] sub-shards done (rcA=$RA rcB=$RB)"

if [ -f "$O/eval/shard_1.pt" ] && [ -f "$O/eval/shard_a.pt" ] && [ -f "$O/eval/shard_b.pt" ]; then
  python method02_advantage_regress_tinyConv/merge_video_shards.py \
    --shards "$O/eval/shard_1.pt" "$O/eval/shard_a.pt" "$O/eval/shard_b.pt" \
    --out "$O/eval/video_curve.json" 2>&1 | tee "$O/eval/merge.log"
  echo "[$(date +%T)] MERGED -> $O/eval/video_curve.json"
else
  echo "[$(date +%T)] missing raw shards: $(ls $O/eval/shard_*.pt 2>/dev/null)"
fi
