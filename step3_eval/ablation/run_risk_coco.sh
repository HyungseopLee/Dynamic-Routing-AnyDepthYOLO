#!/usr/bin/env bash
# Re-measure the KITTI risk-weighted ablation with COCO-exact AP/AR.
# Waits for the main re-measurement (logs_coco/run.log) to finish first so the
# two jobs never share the GPU.
set -u
cd /home/hslee/Desktop/Embedded_AI/context-anydepth-det
source /home/hslee/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate yolov12

LOG=logs_risk/coco.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the main COCO re-measurement to finish ..."
until grep -q "ALL DONE" logs_coco/run.log; do
  if grep -q "FAIL" logs_coco/run.log; then say "main run FAILED -- aborting"; exit 1; fi
  sleep 120
done
say "main run finished; starting risk ablation"

W=results/step2_router/weights/kitti
for w in 1 2 5 10; do
  POL=""
  for s in 0 1 2 3 4; do POL="${POL:+$POL,}s${s}=$W/router_g2x2_both_riskw${w}_s${s}.pt"; done
  say "START eval w=$w"
  python -m step3_eval.eval_video \
      --dataset kitti --weight results/step1_finetune/weights/kitti/best.pt \
      --policies "$POL" \
      --val_cache results/step2_router/cache/kitti/cache_val_g2x2_both.pt \
      --grid 2 --imgsz 384 1248 --conf 0.001 --max_det 100 \
      --budgets 10,20,30,40,50,60,70,80,90 \
      --out results/step3_eval/kitti/eval/video_curve_coco_riskw${w}.json \
      >>"$LOG" 2>&1 && say "OK w=$w" || { say "FAIL w=$w"; exit 1; }
done
say "===== RISK ABLATION DONE ====="
