#!/usr/bin/env bash
# Train BDD100K routers for prev-action ablation: feat=both, p in {0.0, 1.0}, seeds 0-4.
# (p=0.5 already exists as router_g2x2_both_s{0-4}.pt)
# Then eval video on BDD val, producing video_curve_prevp_both_bdd.json for Fig 11.
#
# Usage (from repo root):
#   bash step3_eval/ablation/run_bdd_prevaction_abl.sh

set -e
cd "$(dirname "$0")/../.."   # repo root

PY=$(which python3)
OUT=results/step2_router/weights/bdd100k
CACHE_DIR=results/step2_router/cache/bdd100k
EVAL_DIR=results/step3_eval/bdd100k
CACHE=$CACHE_DIR/cache_train_g2x2_both.pt
VAL=$CACHE_DIR/cache_val_g2x2_both.pt

echo "=== [1/2] Train: feat=both, prev_p in {0.0, 1.0}, seeds 0-4 ==="
for PP in 0.0 1.0; do
  TAG="p$(echo $PP | tr -d '.')"   # p00 / p10
  for S in 0 1 2 3 4; do
    PT="$OUT/router_g2x2_both_s${S}_prevp${TAG}.pt"
    if [ -f "$PT" ]; then
      echo "  [skip] $PT exists"
      continue
    fi
    echo "  training prev_p=$PP seed=$S -> $PT"
    $PY -m step2_train_router.train_policy \
      --dataset bdd100k \
      --cache    "$CACHE" \
      --val_cache "$VAL" \
      --epochs 150 --batch 256 --lr 1e-3 \
      --feat both --norm batch --dropout 0.0 \
      --group_dim 256 --path_dim 8 --hidden 512 \
      --weight_decay 1e-4 \
      --select val_corr \
      --prev_p $PP --seed $S \
      --out "$PT" \
      --tag "both_prevp${TAG}_s${S}"
  done
done

echo ""
echo "=== [2/2] Video eval: all three p values ==="
# p=0.5: existing router_g2x2_both_s0~4
POLS=""
for S in 0 1 2 3 4; do
  POLS+="both_p05_s${S}=$OUT/router_g2x2_both_s${S}.pt,"
done
for PP in 0.0 1.0; do
  TAG="p$(echo $PP | tr -d '.')"
  for S in 0 1 2 3 4; do
    POLS+="both_${TAG}_s${S}=$OUT/router_g2x2_both_s${S}_prevp${TAG}.pt,"
  done
done
POLS="${POLS%,}"

OUT_JSON=$EVAL_DIR/eval/video_curve_prevp_both_bdd.json
echo "  out: $OUT_JSON"

$PY -m step3_eval.eval_video --dataset bdd100k \
  --weight results/step1_finetune/weights/bdd100k/best.pt \
  --policies "$POLS" \
  --grid 2 \
  --conf 0.25 \
  --router_only \
  --val_cache "$VAL" \
  --out "$OUT_JSON"

echo ""
echo "=== Done. JSON: $OUT_JSON ==="
