#!/bin/bash
# Policy learning for AnyDepth-YOLO depth-config decision network.
#
# Usage:
#   bash train_policy_adn.sh cache   # Stage 1: precompute (state, L_super, L_base, FLOPs)
#   bash train_policy_adn.sh train   # Stage 2: RL training of policy network
# bash train_policy_adn.sh cache && bash train_policy_adn.sh train

set -e

# --- Pretrained AnyDepth-YOLO checkpoint (frozen during policy training) ---
WEIGHT=./runs/bdd100k/detect/anydepth-yolov12s/50e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_orig/weights/best.pt
DATA=bdd100k.yaml
IMGSZ=1280
# NOTE: per-sample inner forward dominates step time, so doubling BATCH only
# stabilizes advantage normalization without hurting throughput materially.
BATCH=32
WORKERS=4

# --- Single project root: cache + train results live together ---
PROJECT=./runs/bdd100k/detect/anydepth-yolov12s/50e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_orig/router
CACHE_DIR=$PROJECT/cache
# New output dir for the Lagrangian-reward experiment (preserves the prior run).
# v2: lower entropy bonus + higher LR to let the policy actually commit (H was
# stuck at 5.0+ in v1, indicating c_e=0.10 over-regularized).
OUT_DIR=$PROJECT/train_lagrangian_v2

mkdir -p "$CACHE_DIR" "$OUT_DIR"

STAGE=${1:-cache}

case "$STAGE" in
  cache)
    CUDA_VISIBLE_DEVICES=0 python train_policy_adn.py \
      --stage cache \
      --weight "$WEIGHT" \
      --data "$DATA" \
      --imgsz $IMGSZ --batch $BATCH --workers $WORKERS \
      --cache-dir "$CACHE_DIR" \
      2>&1 | tee "$CACHE_DIR/cache.log"
    ;;
  train)
    CUDA_VISIBLE_DEVICES=0 python train_policy_adn.py \
      --stage train \
      --weight "$WEIGHT" \
      --data "$DATA" \
      --imgsz $IMGSZ --batch $BATCH --workers $WORKERS \
      --cache-dir "$CACHE_DIR" \
      --out-dir   "$OUT_DIR" \
      --epochs 5 \
      --lr 3e-4 \
      --hidden 256 \
      --lambda-flops 1.0 \
      --c_v 0.5 --c_e 0.02 \
      --reward-clip 3.0 \
      2>&1 | tee "$OUT_DIR/train.log"
    ;;
  *)
    echo "Usage: bash train_policy_adn.sh [cache|train]"
    exit 1
    ;;
esac
