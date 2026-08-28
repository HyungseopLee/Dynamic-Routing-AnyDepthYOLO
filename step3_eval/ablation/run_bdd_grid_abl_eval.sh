#!/usr/bin/env bash
# Video eval for Fig 10 (BDD100K router grid/arch ablation).
# Run 1 (grid=2): TinyConv 2x2 + GAP-MLP
# Run 2 (grid=8): TinyConv 8x8
# Then merge all shards into one JSON.
#
# Usage (from repo root):
#   bash step3_eval/ablation/run_bdd_grid_abl_eval.sh

set -e
cd "$(dirname "$0")/../.."

O=results/step2_router/weights/bdd100k
CACHE_DIR=results/step2_router/cache/bdd100k
EVAL_DIR=results/step3_eval/bdd100k
W=results/step1_finetune/weights/bdd100k/best.pt
mkdir -p "$EVAL_DIR/eval"

# TinyConv 2x2 (router_bn*) and GAP-MLP (policy_gmbn*) already exist in
# video_curve_archabl.json — no need to re-eval. Only grid=8 is new.

# ── [1/2] grid=8: TinyConv 8x8 ───────────────────────────────────────────────
POL_G8=""
for S in 0 1 2 3 4; do
  POL_G8+="tinyconv_g8_s${S}=$O/router_g8x8_both_s${S}.pt,"
done
POL_G8="${POL_G8%,}"

echo ""
echo "=== [1/2] Video eval grid=8 (TinyConv 8x8) ==="
python -m step3_eval.eval_video --dataset bdd100k \
    --weight "$W" \
    --policies "$POL_G8" \
    --grid 8 --imgsz 720 1280 --conf 0.25 \
    --val_cache "$CACHE_DIR/cache_val_g8x8_both.pt" \
    --budgets 1,5,10,20,30,40,50,60,70,80,90,95,99 \
    --num_shards 1 --shard_id 0 \
    --raw_out "$EVAL_DIR/eval/grid_abl_g8_shard_0.pt" \
    2>&1 | tee "$EVAL_DIR/eval/grid_abl_g8.log"

# ── [3/3] Merge ───────────────────────────────────────────────────────────────
echo ""
echo "=== [2/2] Merging: g8 shard + existing archabl (g2 + gapmpl) ==="
python -m step3_eval.merge_video_shards \
    --shards "$EVAL_DIR/eval/grid_abl_g8_shard_0.pt" \
    --out "$EVAL_DIR/eval/video_curve_router_grid_bdd.json"

echo ""
echo "=== Done: $EVAL_DIR/eval/video_curve_router_grid_bdd.json ==="
