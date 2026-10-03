#!/usr/bin/env bash
# Risk-weighted loss experiment (reviewer response), KITTI only.
#
#   L_risk(w) = L(all GT) + (w - 1) * L(safety-critical GT only)
#   safety-critical KITTI classes: 3 pedestrian, 4 person_sitting, 5 cyclist
#
# The cache stores loss and loss_sc separately, so one rebuild covers every w;
# train_policy.py --risk_weight w composes the advantage at train time.
set -u
cd /home/hslee/Desktop/Embedded_AI/context-anydepth-det
source /home/hslee/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate yolov12

LOG=logs_risk/run.log   # relative to repo root (script cds there)
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run() { say "START $1"; shift; "$@" >>"$LOG" 2>&1 && say "OK" || { say "FAIL (exit $?)"; exit 1; }; }

WEIGHT=results/step1_finetune/weights/kitti/best.pt
CACHE=results/step2_router/cache/kitti
W=results/step2_router/weights/kitti
E=results/step3_eval/kitti/eval
SEEDS="0 1 2 3 4"
SKIP_BUILD=${SKIP_BUILD:-0}
WEIGHTS_W="1 2 5 10"

say "===== risk-weighted loss experiment (KITTI) ====="

# ---- 1. cache rebuild (overwrites, via .tmp so a crash keeps the old file) ----
if [ "$SKIP_BUILD" = 0 ]; then
for split in train val; do
  run "build_cache $split" python -m step2_train_router.build_cache \
      --weight $WEIGHT --data ultralytics/cfg/datasets/kitti.yaml --dataset kitti \
      --split $split --imgsz 384 1248 --feat both --grid 2 --batch 16 \
      --risk_classes 3 4 5 \
      --out $CACHE/cache_${split}_g2x2_both.tmp.pt
  mv $CACHE/cache_${split}_g2x2_both.tmp.pt $CACHE/cache_${split}_g2x2_both.pt
  say "cache_${split} replaced"
done
fi

# ---- 2. train routers: w=1 is the plain-advantage baseline ----
if [ "$SKIP_BUILD" = 0 ]; then
for w in $WEIGHTS_W; do
  for s in $SEEDS; do
    run "train w=$w seed=$s" python -m step2_train_router.train_policy \
        --dataset kitti \
        --cache $CACHE/cache_train_g2x2_both.pt \
        --val_cache $CACHE/cache_val_g2x2_both.pt \
        --feat both --arch tinyconv --hidden 128 --group_dim 64 --path_dim 8 \
        --epochs 300 --batch 256 --lr 1e-3 --norm batch \
        --select val_corr --regress_loss mse --prev_p 0.5 \
        --risk_weight $w --seed $s \
        --out $W/router_g2x2_both_riskw${w}_s${s}.pt
  done
done
fi

# ---- 3. eval each w at COCO-standard conf 0.001, 5 seeds averaged ----
for w in $WEIGHTS_W; do
  POL=""
  for s in $SEEDS; do POL="${POL:+$POL,}s${s}=$W/router_g2x2_both_riskw${w}_s${s}.pt"; done
  run "eval w=$w" python -m step3_eval.eval_video \
      --dataset kitti --weight $WEIGHT \
      --policies "$POL" \
      --val_cache $CACHE/cache_val_g2x2_both.pt \
      --grid 2 --imgsz 384 1248 --conf 0.001 \
      --budgets 10,20,30,40,50,60,70,80,90 \
      --out $E/video_curve_riskw${w}.json
done

say "===== ALL DONE ====="
