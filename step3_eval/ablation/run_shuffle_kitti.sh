#!/usr/bin/env bash
# Temporal-coherence stress test (KITTI): shuffle each sequence's frame order so
# the recursive routing signal always comes from an unrelated scene -- every frame
# behaves like a hard scene cut. Frames are still scored against their own GT.
#
# The bar is the random baseline at the same budget, measured in the same run:
# any policy that spends budget beats always_base, so always_base proves nothing.
# Waits for the risk ablation to finish so the jobs never share the GPU.
set -u
cd /home/hslee/Desktop/Embedded_AI/context-anydepth-det
source /home/hslee/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate yolov12

LOG=logs_shuffle/run.log
mkdir -p logs_shuffle
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the risk ablation to finish ..."
until grep -q "RISK ABLATION DONE" logs_risk/coco.log 2>/dev/null; do
  if grep -q "FAIL" logs_risk/coco.log 2>/dev/null; then say "upstream FAILED -- aborting"; exit 1; fi
  sleep 120
done
say "GPU free; starting shuffle stress test"

W=results/step2_router/weights/kitti
POL=""
for s in 0 1 2 3 4; do POL="${POL:+$POL,}both_s${s}=$W/router_g2x2_both_s${s}.pt"; done

for sh in 0 1 2; do
  say "START shuffle_seed=$sh"
  python -m step3_eval.eval_video \
      --dataset kitti --weight results/step1_finetune/weights/kitti/best.pt \
      --policies "$POL" \
      --val_cache results/step2_router/cache/kitti/cache_val_g2x2_both.pt \
      --grid 2 --imgsz 384 1248 --conf 0.001 --max_det 100 \
      --budgets 10,20,30,40,50,60,70,80,90 \
      --shuffle_frames --shuffle_seed "$sh" \
      --out "results/step3_eval/kitti/eval/video_curve_shuffle_s${sh}.json" \
      >>"$LOG" 2>&1 && say "OK shuffle_seed=$sh" || { say "FAIL shuffle_seed=$sh"; exit 1; }
done
say "===== SHUFFLE DONE ====="
