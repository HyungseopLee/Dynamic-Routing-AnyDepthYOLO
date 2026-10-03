#!/usr/bin/env bash
# Causal-routing study (KITTI), then the Waymo main result.
#
# --causal decides frame t from the A-hat produced while executing frame t-1, which
# is what the deployed controller does. Combined with --shuffle_frames it gives the
# real temporal-coherence stress test: the routing signal then comes from an
# unrelated scene, so the policy should degrade towards random.
#
#   causal + normal order   -> deployment conditions
#   causal + shuffled order -> routing signal destroyed
#
# Waits for the Waymo main result to finish (the main figures have priority),
# then runs the four KITTI evals.
set -u
cd /home/hslee/Desktop/Embedded_AI/context-anydepth-det
source /home/hslee/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate yolov12

LOG=logs_causal/run.log
mkdir -p logs_causal
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the Waymo main result (main figures come first) ..."
# Wait for the GPU to be free rather than for a success marker: a failed Waymo run
# is usually restarted by hand, and aborting on "FAIL" left this job dead overnight.
until ! pgrep -f "eval_video --dataset waymo" >/dev/null; do sleep 120; done
say "no Waymo job running"
say "waymo done; GPU free"

W=results/step2_router/weights/kitti
POL=""
for s in 0 1 2 3 4; do POL="${POL:+$POL,}both_s${s}=$W/router_g2x2_both_s${s}.pt"; done

run_kitti() {   # out_name  extra args...
  local out="$1"; shift
  say "START $out"
  python -m step3_eval.eval_video \
      --dataset kitti --weight results/step1_finetune/weights/kitti/best.pt \
      --data_root /media/data/kitti-tracking \
      --policies "$POL" \
      --val_cache results/step2_router/cache/kitti/cache_val_g2x2_both.pt \
      --grid 2 --imgsz 384 1248 --conf 0.001 --max_det 100 --random_seeds 5 \
      --budgets 10,20,30,40,50,60,70,80,90 "$@" \
      --out "results/step3_eval/kitti/eval/${out}.json" >>"$LOG" 2>&1 \
      && say "OK $out" || { say "FAIL $out"; return 1; }
}

run_kitti video_curve_causal --causal
for sh in 0 1 2 3 4; do
  run_kitti "video_curve_causal_shuffle_s${sh}" --causal --shuffle_frames --shuffle_seed "$sh"
done
say "===== CAUSAL STUDY DONE ====="
