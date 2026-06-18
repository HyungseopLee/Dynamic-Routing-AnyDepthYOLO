#!/bin/bash
# Train 5-seed feat=BOTH depth-routing policies on the Waymo both-cache.
# Same recipe as feat=input (batch-norm, regress/mse, select val_corr) but the
# tinyConv now also consumes the neck pred grids -> policy_both_{seed}.pt.
set -uo pipefail
source ~/anaconda3/bin/activate yolov12 2>/dev/null
cd /home/hslee/context-anydepth-det/anydepth-yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

O=method02_advantage_regress_tinyConv/outputs/waymo
LOGDIR=$O/logs
mkdir -p "$LOGDIR"

COMMON="--dataset waymo --feat both --norm batch \
        --cache $O/cache_train_both.pt --val_cache $O/cache_val_both.pt \
        --epochs 50 --select val_corr --mode regress --regress_loss mse \
        --device cuda:0"

for S in 0 1 2 3 4; do
  echo "[$(date +%T)] train BOTH policy seed $S -> $O/policy_both_${S}.pt"
  python method02_advantage_regress_tinyConv/train_policy.py $COMMON \
    --seed $S --tag both_seed$S \
    --logdir "$LOGDIR" \
    --out "$O/policy_both_${S}.pt" \
    2>&1 | tee "$LOGDIR/policy_both_${S}.log"
done
echo "[$(date +%T)] both policies done"
