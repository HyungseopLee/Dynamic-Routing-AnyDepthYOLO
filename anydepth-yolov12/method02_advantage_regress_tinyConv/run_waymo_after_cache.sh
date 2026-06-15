#!/bin/bash
# Orchestrator: wait for the Waymo caches (built by run_waymo_cache.sh) to finish,
# then run 5-seed policy training, then the video eval + PI. Detached-safe.
set -uo pipefail
cd /home/hslee/context-anydepth-det/anydepth-yolov12
DIR=method02_advantage_regress_tinyConv
O=$DIR/outputs/waymo
LOG=$O/logs/orchestrator.log
mkdir -p "$O/logs"
echo "[$(date +%T)] orchestrator start; waiting for caches" | tee "$LOG"

# wait until both caches exist AND the builder has exited (file size stable)
until [ -f "$O/cache_train.pt" ] && ! pgrep -f "build_cache.py.*--dataset waymo" >/dev/null; do
  sleep 30
done
echo "[$(date +%T)] caches ready:" | tee -a "$LOG"
ls -lh "$O"/cache_*.pt | tee -a "$LOG"

echo "[$(date +%T)] === policy training ===" | tee -a "$LOG"
bash $DIR/run_waymo_policy.sh 2>&1 | tee -a "$LOG"

echo "[$(date +%T)] === video eval + PI ===" | tee -a "$LOG"
bash $DIR/run_waymo_eval.sh 2>&1 | tee -a "$LOG"

echo "[$(date +%T)] orchestrator done" | tee -a "$LOG"
