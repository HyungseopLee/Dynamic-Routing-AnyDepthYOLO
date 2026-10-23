#!/usr/bin/env bash
# Main results for the three datasets, with COCO-exact AP/AR.
#
#   - compute_ap reproduces pycocotools COCOeval.accumulate()
#   - --max_det 100 caps detections per (frame, class), as pycocotools does
#   - the random baseline runs 5 seeds, so it gets the same mean+/-std treatment
#     as the router instead of a single noisy realisation
#   - each run cross-checks the two const strategies against pycocotools
#
# Routers, seeds, conf and budgets are unchanged from the paper's configuration.
set -u
cd /home/hslee/Desktop/Embedded_AI/context-anydepth-det
source /home/hslee/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate yolov12

LOG=logs_coco/run.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

pols() {
  local p=""
  for s in 0 1 2 3 4; do
    p="${p:+$p,}both_s${s}=results/step2_router/weights/$1/router_g2x2_both_s${s}.pt"
  done
  echo "$p"
}

eval_one() {   # dataset  h  w  out_dir  data_root
  local ds="$1" h="$2" w="$3" od="$4" root="$5"
  say "START $ds"
  python -m step3_eval.eval_video \
      --dataset "$ds" --weight "results/step1_finetune/weights/$ds/best.pt" \
      --data_root "$root" \
      --policies "$(pols "$ds")" \
      --val_cache "results/step2_router/cache/$ds/cache_val_g2x2_both.pt" \
      --grid 2 --imgsz "$h" "$w" --conf 0.001 --max_det 100 --random_seeds 5 \
      --budgets 10,20,30,40,50,60,70,80,90 \
      --coco_out "$od/coco_check" --coco_strategies always_base,always_super \
      --out "$od/video_curve_coco.json" >>"$LOG" 2>&1 \
      && say "OK $ds" || { say "FAIL $ds"; return 1; }
}

say "===== main results: COCO-exact AP/AR, 5-seed random ====="
eval_one kitti    384 1248 results/step3_eval/kitti/eval      /media/data/kitti-tracking
eval_one bdd100k  720 1280 results/step3_eval/bdd100k/eval    /media/data/bdd100k_mot/val
eval_one waymo   1280 1920 results/step3_eval/waymo/eval_both /media/data/waymo_yolo/val
say "===== ALL DONE ====="
