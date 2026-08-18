#!/usr/bin/env bash
# PI controller ablation experiments — k_p and k_i sensitivity at α=0.85
#
# Runs 4 experiments sequentially. Each produces a JSON + PDF in:
#   method_advantage_regress/outputs/bdd100k/
#
# Usage (from anydepth-yolov12/ — the project root):
#   sudo jetson_clocks
#   bash method_advantage_regress/jetson/run_ablation_experiments.sh 2>&1 | tee ablation_run.log

set -uo pipefail   # -e removed: continue to next experiment even on OOM/crash

BASE="trt_bench/onnx/bdd_pooled/base.fp16.engine"
SUPER="trt_bench/onnx/bdd_pooled/super.fp16.engine"
ROUTER_ENG="trt_bench/onnx/bdd_pooled/router.fp16.engine"
ROUTER="method_advantage_regress/outputs/bdd100k/policy_both_0.pt"

COMMON="--base $BASE --super $SUPER --router_engine $ROUTER_ENG --router $ROUTER \
        --mode latency --warmup 60 --win 30"

EXPERIMENTS=(
    "--kp 0.5  --ki 0.33 --beta 0.85"   # k_p low
    "--kp 5.0  --ki 0.33 --beta 0.85"   # k_p high
    "--kp 2.0  --ki 0.10 --beta 0.85"   # k_i low
    "--kp 2.0  --ki 1.00 --beta 0.85"   # k_i high
)
LABELS=(
    "k_p=0.5  k_i=0.33  alpha=0.85"
    "k_p=5.0  k_i=0.33  alpha=0.85"
    "k_p=2.0  k_i=0.10  alpha=0.85"
    "k_p=2.0  k_i=1.00  alpha=0.85"
)

TOTAL=${#EXPERIMENTS[@]}
FAILED=()

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== PI ablation: $TOTAL experiments ==="
log "Engines: $BASE"

# Drop page cache before starting to maximise available memory
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
log "Page cache dropped. Free memory: $(free -h | awk '/^Mem/{print $4}')"

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    LBL="${LABELS[$i]}"
    IDX=$((i + 1))

    log "--- Experiment $IDX/$TOTAL: $LBL ---"

    # Drop cache between runs to avoid OOM from leftover GPU allocations
    if [ $IDX -gt 1 ]; then
        sudo sync
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
        sleep 2
        log "Cache dropped. Free: $(free -h | awk '/^Mem/{print $4}')"
    fi

    if PYTHONPATH=. python3 \
           method_advantage_regress/jetson/online_budget_demo_stream.py \
           $COMMON $EXP; then
        log "Experiment $IDX/$TOTAL OK."
    else
        log "Experiment $IDX/$TOTAL FAILED (exit $?). Continuing..."
        FAILED+=("$LBL")
    fi
done

log "=== Summary ==="
if [ ${#FAILED[@]} -eq 0 ]; then
    log "All $TOTAL experiments succeeded."
else
    log "${#FAILED[@]} experiment(s) failed:"
    for f in "${FAILED[@]}"; do log "  FAILED: $f"; done
fi

log "JSONs in: method_advantage_regress/outputs/bdd100k/"
ls -lh method_advantage_regress/outputs/bdd100k/jetson_trtengine_720*b0.85*.json 2>/dev/null || true
