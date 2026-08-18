#!/usr/bin/env bash
# Fig 10 (BDD100K): Router architecture & grid ablation, cache-based training.
#
# TinyConv 2x2 (default): existing router_both_0-4.pt  (already done)
# GAP-MLP      (feat=both): spatial-mean the existing cache_train_both.pt  -> train
# TinyConv 8x8 (feat=both): build cache_train_g8_both.pt (~50GB) -> train
#
# Usage (from repo root):
#   bash step3_eval/ablation/run_bdd_router_grid_abl.sh

set -e
cd "$(dirname "$0")/../.."

OUT=results/step2_router/weights/bdd100k
CACHE_DIR=results/step2_router/cache/bdd100k
EVAL_DIR=results/step3_eval/bdd100k
WEIGHT=results/step1_finetune/weights/bdd100k/best.pt
DATA=ultralytics/cfg/datasets/bdd100k.yaml
CACHE_BOTH=$CACHE_DIR/cache_train_both.pt
VAL_BOTH=$CACHE_DIR/cache_val_both.pt
CACHE_G8=$CACHE_DIR/cache_train_g8_both.pt
VAL_G8=$CACHE_DIR/cache_val_g8_both.pt

# ── [1/3] GAP-MLP: reuse existing cache_train_both.pt ────────────────────────
echo "=== [1/3] Train GAP-MLP (feat=both, arch=gapmpl) seeds 0-4 ==="
for S in 0 1 2 3 4; do
  PT="$OUT/router_gapmpl_both_s${S}.pt"
  if [ -f "$PT" ]; then echo "  [skip] $PT"; continue; fi
  echo "  GAP-MLP seed=$S"
  python -m step2_train_router.train_policy \
    --dataset bdd100k \
    --cache    "$CACHE_BOTH" \
    --val_cache "$VAL_BOTH" \
    --epochs 30 --batch 256 --lr 1e-3 \
    --feat both --arch gapmpl \
    --norm batch --dropout 0.0 \
    --weight_decay 1e-4 \
    --select val_corr \
    --prev_p 0.5 --seed $S \
    --out "$PT" \
    --tag "gapmpl_both_s${S}"
done

# ── [2/3] Build TinyConv 8x8 cache (~50 GB) ──────────────────────────────────
echo ""
echo "=== [2/3] Build cache grid=8 (feat=both) ==="
if [ ! -f "$CACHE_G8" ]; then
  python -m step2_train_router.build_cache \
    --weight  "$WEIGHT" \
    --data    "$DATA" \
    --dataset bdd100k \
    --split   train \
    --feat    both \
    --grid    8 \
    --batch   8 \
    --fp16 \
    --imgsz   720 1280 \
    --out     "$CACHE_G8"
else
  echo "  [skip] $CACHE_G8 exists"
fi
if [ ! -f "$VAL_G8" ]; then
  python -m step2_train_router.build_cache \
    --weight  "$WEIGHT" \
    --data    "$DATA" \
    --dataset bdd100k \
    --split   val \
    --feat    both \
    --grid    8 \
    --batch   8 \
    --fp16 \
    --imgsz   720 1280 \
    --out     "$VAL_G8"
else
  echo "  [skip] $VAL_G8 exists"
fi

# ── [3/3] Train TinyConv 8x8 ─────────────────────────────────────────────────
echo ""
echo "=== [3/3] Train TinyConv 8x8 (feat=both) seeds 0-4 ==="
for S in 0 1 2 3 4; do
  PT="$OUT/router_tinyconv_g8_both_s${S}.pt"
  if [ -f "$PT" ]; then echo "  [skip] $PT"; continue; fi
  echo "  TinyConv 8x8 seed=$S"
  python -m step2_train_router.train_policy \
    --dataset bdd100k \
    --cache    "$CACHE_G8" \
    --val_cache "$VAL_G8" \
    --epochs 30 --batch 256 --lr 1e-3 \
    --feat both --arch tinyconv \
    --group_dim 256 --path_dim 8 --hidden 512 \
    --norm batch --dropout 0.0 \
    --weight_decay 1e-4 \
    --select val_corr \
    --prev_p 0.5 --seed $S \
    --out "$PT" \
    --tag "tinyconv_g8_both_s${S}"
done

echo ""
echo "=== All training done. Run video eval next. ==="
