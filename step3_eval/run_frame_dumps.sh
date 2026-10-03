#!/usr/bin/env bash
# One GPU pass per dataset that records, for every frame, the matched detections of
# BOTH paths plus the router outputs (step3_eval/eval_video.py --frame_dump).
#
# With that dump, any per-frame routing policy can be scored offline in milliseconds:
# the exact tau that realises 10%, 20%, ... SUPER usage (step3_eval/ablation/tau_sweep.py),
# the random baseline at matched usage with as many seeds as wanted, causal vs
# non-causal routing, and the scene-change guard. None of them need the GPU again.
#
# The dump itself is policy-agnostic, so the strategy list here is kept minimal.
set -u
cd /home/hslee/Desktop/Embedded_AI/context-anydepth-det
source /home/hslee/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate yolov12
# unbuffered: progress lines reach the log as they happen, not in 8 KB chunks
export PYTHONUNBUFFERED=1

LOG=logs_dump/run.log
mkdir -p logs_dump
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ "${WAIT_FOR_GUARD:-1}" = 1 ]; then
  say "waiting for the pixel-guard run ..."
  until grep -q "GUARD DONE" logs_guard/run.log 2>/dev/null; do
    if grep -q "FAIL" logs_guard/run.log 2>/dev/null; then say "guard FAILED -- continuing anyway"; break; fi
    sleep 60
  done
fi

pols() {
  local p=""
  for s in 0 1 2 3 4; do
    p="${p:+$p,}s${s}=results/step2_router/weights/$1/router_g2x2_both_s${s}.pt"
  done
  echo "$p"
}

dump_one() {   # dataset h w out_dir data_root
  local ds="$1" h="$2" w="$3" od="$4" root="$5"
  say "START $ds"
  python -m step3_eval.eval_video \
      --dataset "$ds" --weight "results/step1_finetune/weights/$ds/best.pt" \
      --data_root "$root" --policies "$(pols "$ds")" \
      --grid 2 --imgsz "$h" "$w" --conf 0.001 --max_det 100 \
      --router_only --router_taus 2 --random_seeds 0 \
      --frame_dump "$od/frames.npz" --out "$od/video_curve_dumprun.json" \
      >>"$LOG" 2>&1 && say "OK $ds" || { say "FAIL $ds"; return 1; }
}

say "===== frame dumps ====="
dump_one kitti    384 1248 results/step3_eval/kitti/eval      /media/data/kitti-tracking
dump_one bdd100k  720 1280 results/step3_eval/bdd100k/eval    /media/data/bdd100k_mot/val
dump_one waymo   1280 1920 results/step3_eval/waymo/eval_both /media/data/waymo_yolo/val
say "===== DUMPS DONE ====="
