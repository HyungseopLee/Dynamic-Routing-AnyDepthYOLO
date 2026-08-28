#!/usr/bin/env bash
# KITTI grid ablation: train G=1x1 (GAP-MLP) and G=8x8 (TinyConv) with feat=both.
# G=2x2 (feat=both) already exists: router_g2x2_both_s0~4.pt
#
# All three share the G=2x2 recipe so the grid size is the only variable:
#   feat=both, epochs=300, batch=256, lr=1e-3, group_dim=64, hidden=128,
#   path_dim=8, norm=batch, dropout=0, weight_decay=0, prev_p=0.5,
#   regress_loss=mse (direct regression on the advantage), select=val_corr.
#
# Usage (from repo root):
#   bash step3_eval/ablation/run_kitti_grid_abl.sh

set -e
cd "$(dirname "$0")/../.."

PY=${PY:-python}
OUT=results/step2_router/weights/kitti
CACHE_DIR=results/step2_router/cache/kitti
EVAL_DIR=results/step3_eval/kitti
WEIGHT=results/step1_finetune/weights/kitti/best.pt
DATA=ultralytics/cfg/datasets/kitti.yaml

CACHE_G2_TRAIN=$CACHE_DIR/cache_train_g2x2_both.pt   # feat=both already built
CACHE_G2_VAL=$CACHE_DIR/cache_val_g2x2_both.pt
CACHE_G8_TRAIN=$CACHE_DIR/cache_train_g8x8_both.pt
CACHE_G8_VAL=$CACHE_DIR/cache_val_g8x8_both.pt

# ── [1/4] Train G=1x1 GAP-MLP (feat=both, reuse cache_train_g2x2_both.pt) ───────────
echo ""
echo "=== [1/4] Train GAP-MLP G=1x1 (feat=both) seeds 0-4 ==="
for S in 0 1 2 3 4; do
  PT="$OUT/router_gapmlp_g1x1_both_s${S}.pt"
  if [ -f "$PT" ]; then echo "  [skip] $PT"; continue; fi
  echo "  GAP-MLP seed=$S"
  "$PY" -m step2_train_router.train_policy \
    --dataset kitti \
    --cache    "$CACHE_G2_TRAIN" \
    --val_cache "$CACHE_G2_VAL" \
    --epochs 300 --batch 256 --lr 1e-3 \
    --feat both --arch gapmpl \
    --group_dim 64 --hidden 128 \
    --norm batch --dropout 0.0 \
    --weight_decay 0.0 \
    --regress_loss mse \
    --select val_corr \
    --prev_p 0.5 --seed $S \
    --out "$PT" \
    --logdir "$OUT/ablation/logs" \
    --tag "gapmpl_both_s${S}"
done

# ── [2/4] Build G=8x8 cache (feat=both) ──────────────────────────────────────
echo ""
echo "=== [2/4] Build G=8x8 cache (feat=both) ==="
if [ ! -f "$CACHE_G8_TRAIN" ]; then
  "$PY" -m step2_train_router.build_cache \
    --weight  "$WEIGHT" \
    --data    "$DATA" \
    --dataset kitti \
    --split   train \
    --feat    both \
    --grid    8 \
    --batch   16 \
    --imgsz   384 1248 \
    --out     "$CACHE_G8_TRAIN"
else
  echo "  [skip] $CACHE_G8_TRAIN exists"
fi

if [ ! -f "$CACHE_G8_VAL" ]; then
  "$PY" -m step2_train_router.build_cache \
    --weight  "$WEIGHT" \
    --data    "$DATA" \
    --dataset kitti \
    --split   val \
    --feat    both \
    --grid    8 \
    --batch   16 \
    --imgsz   384 1248 \
    --out     "$CACHE_G8_VAL"
else
  echo "  [skip] $CACHE_G8_VAL exists"
fi

# ── [3/4] Train G=8x8 TinyConv (feat=both) ───────────────────────────────────
echo ""
echo "=== [3/4] Train TinyConv G=8x8 (feat=both) seeds 0-4 ==="
for S in 0 1 2 3 4; do
  PT="$OUT/router_g8x8_both_s${S}.pt"
  if [ -f "$PT" ]; then echo "  [skip] $PT"; continue; fi
  echo "  TinyConv G=8x8 seed=$S"
  "$PY" -m step2_train_router.train_policy \
    --dataset kitti \
    --cache    "$CACHE_G8_TRAIN" \
    --val_cache "$CACHE_G8_VAL" \
    --epochs 300 --batch 256 --lr 1e-3 \
    --feat both --arch tinyconv \
    --group_dim 64 --path_dim 8 --hidden 128 \
    --norm batch --dropout 0.0 \
    --weight_decay 0.0 \
    --regress_loss mse \
    --select val_corr \
    --prev_p 0.5 --seed $S \
    --out "$PT" \
    --logdir "$OUT/ablation/logs" \
    --tag "tinyconv_g8_both_s${S}"
done

# ── [4/4] Video eval ─────────────────────────────────────────────────────────
echo ""
echo "=== [4/4] Video eval: G=1x1, G=2x2, G=8x8 ==="

POL_GAP=""
for S in 0 1 2 3 4; do
  POL_GAP+="gap_s${S}=$OUT/router_gapmlp_g1x1_both_s${S}.pt,"
done
POL_GAP="${POL_GAP%,}"

POL_G2=""
for S in 0 1 2 3 4; do
  POL_G2+=",bn${S}=$OUT/router_g2x2_both_s${S}.pt"
done
POL_G2="${POL_G2#,}"

POL_G8=""
for S in 0 1 2 3 4; do
  POL_G8+=",g8_s${S}=$OUT/router_g8x8_both_s${S}.pt"
done
POL_G8="${POL_G8#,}"

# eval G=1x1 and G=2x2 (both use grid=2 cache)
"$PY" -m step3_eval.eval_video --dataset kitti \
    --weight "$WEIGHT" \
    --policies "${POL_GAP},${POL_G2}" \
    --grid 2 --imgsz 384 1248 --conf 0.25 \
    --val_cache "$CACHE_G2_VAL" \
    --budgets 10,20,30,40,50,60,70,80,90 \
    --out "$EVAL_DIR/eval/video_curve_grid_abl_g1_g2.json" \
    2>&1 | tee "$EVAL_DIR/eval/video_curve_grid_abl_g1_g2.log"

# eval G=8x8
"$PY" -m step3_eval.eval_video --dataset kitti \
    --weight "$WEIGHT" \
    --policies "$POL_G8" \
    --grid 8 --imgsz 384 1248 --conf 0.25 \
    --val_cache "$CACHE_G8_VAL" \
    --budgets 10,20,30,40,50,60,70,80,90 \
    --out "$EVAL_DIR/eval/video_curve_grid_abl_g8.json" \
    2>&1 | tee "$EVAL_DIR/eval/video_curve_grid_abl_g8.log"

echo ""
echo "=== All done. Run python -m paper_figures.plot_ablation_router_grid --dataset kitti to regenerate figure. ==="
