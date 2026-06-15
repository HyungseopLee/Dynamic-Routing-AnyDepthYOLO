#!/bin/bash
# Train 5-seed depth-routing policies on the Waymo cache, mirroring the BDD recipe
# (feat=input backbone tinyConv, batch-norm, regress/mse, select by val_corr).
# Caches are tiny tensors -> 1 GPU is plenty; just loop the 5 seeds.
set -uo pipefail
source ~/anaconda3/bin/activate yolov12 2>/dev/null
cd /home/hslee/context-anydepth-det/anydepth-yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

O=method02_advantage_regress_tinyConv/outputs/waymo
LOGDIR=$O/logs
mkdir -p "$LOGDIR"

COMMON="--dataset waymo --feat input --norm batch \
        --epochs 50 --select val_corr --mode regress --regress_loss mse \
        --device cuda:0"

for S in 0 1 2 3 4; do
  echo "[$(date +%T)] train policy seed $S -> $O/policy_${S}.pt"
  python method02_advantage_regress_tinyConv/train_policy.py $COMMON \
    --seed $S --tag seed$S \
    --logdir "$LOGDIR" \
    --out "$O/policy_${S}.pt" \
    2>&1 | tee "$LOGDIR/policy_${S}.log"
done
echo "[$(date +%T)] policies done"
