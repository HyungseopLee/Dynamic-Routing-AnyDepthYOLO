#!/usr/bin/env bash
# Waymo main-result re-measurement with COCO-exact AP/AR.
# Split out from run_main_coco.sh, which used the wrong default --data_root
# (/media/data/waymo/val) and so evaluated zero sequences.
set -u
cd /home/hslee/Desktop/Embedded_AI/context-anydepth-det
source /home/hslee/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate yolov12

LOG=logs_coco/waymo.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

W=results/step2_router/weights/waymo
POL=""
for s in 0 1 2 3 4; do POL="${POL:+$POL,}both_s${s}=$W/router_g2x2_both_s${s}.pt"; done

say "START waymo"
python -m step3_eval.eval_video \
    --dataset waymo --weight results/step1_finetune/weights/waymo/best.pt \
    --data_root /media/data/waymo_yolo/val \
    --policies "$POL" \
    --val_cache results/step2_router/cache/waymo/cache_val_g2x2_both.pt \
    --grid 2 --imgsz 1280 1920 --conf 0.001 --max_det 100 --random_seeds 5 \
    --budgets 10,20,30,40,50,60,70,80,90 \
    --coco_out results/step3_eval/waymo/eval_both/coco_check \
    --coco_strategies always_base,always_super \
    --out results/step3_eval/waymo/eval_both/video_curve_coco.json \
    >>"$LOG" 2>&1 && say "OK waymo" || { say "FAIL waymo"; exit 1; }
say "===== WAYMO DONE ====="
