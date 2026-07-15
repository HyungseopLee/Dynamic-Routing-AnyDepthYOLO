#!/usr/bin/env bash
# Train BDD100K routers for prev-action ablation: feat=both, p in {0.0, 1.0}, seeds 0-4.
# (p=0.5 already exists as router_both_{0-4}.pt)
# Then eval video on BDD val, producing video_curve_prevp_both_bdd.json for Fig 11.
#
# Usage (from repo root):
#   bash method_advantage_regress/ablation/run_bdd_prevaction_abl.sh

set -e
cd "$(dirname "$0")/../.."   # repo root

PY=$(which python3)
PKG=method_advantage_regress
OUT=$PKG/outputs/bdd100k
CACHE=$PKG/outputs/bdd100k/cache_train_both.pt
VAL=$PKG/outputs/bdd100k/cache_val_both.pt

echo "=== [1/2] Train: feat=both, prev_p in {0.0, 1.0}, seeds 0-4 ==="
for PP in 0.0 1.0; do
  TAG="p$(echo $PP | tr -d '.')"   # p00 / p10
  for S in 0 1 2 3 4; do
    PT="$OUT/router_both_prevp${TAG}_s${S}.pt"
    if [ -f "$PT" ]; then
      echo "  [skip] $PT exists"
      continue
    fi
    echo "  training prev_p=$PP seed=$S -> $PT"
    $PY -m $PKG.train_router \
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
# p=0.5: existing router_both_0~4
POLS=""
for S in 0 1 2 3 4; do
  POLS+="both_p05_s${S}=$OUT/router_both_${S}.pt,"
done
for PP in 0.0 1.0; do
  TAG="p$(echo $PP | tr -d '.')"
  for S in 0 1 2 3 4; do
    POLS+="both_${TAG}_s${S}=$OUT/router_both_prevp${TAG}_s${S}.pt,"
  done
done
POLS="${POLS%,}"

OUT_JSON=$PKG/outputs/bdd100k/eval/video_curve_prevp_both_bdd.json
echo "  out: $OUT_JSON"

$PY -m $PKG.eval_video --dataset bdd100k \
  --weight finetuning_AnyDepthYOLO/weights/bdd100k/anydepth_best.pt \
  --policies "$POLS" \
  --grid 2 \
  --conf 0.25 \
  --router_only \
  --val_cache "$VAL" \
  --out "$OUT_JSON"

echo ""
echo "=== Done. JSON: $OUT_JSON ==="
