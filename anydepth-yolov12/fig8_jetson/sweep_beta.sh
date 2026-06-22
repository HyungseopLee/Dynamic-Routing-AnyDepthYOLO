#!/usr/bin/env bash
# Low-beta sweep: regulate on a faster latency estimate so the controller dithers
# SUPER frames evenly (sigma-delta-like) instead of in advantage-correlated runs,
# which is what inflates the windowed-mean ripple on the large-gap Jetson.
set -e
cd /home/hslee/context-anydepth-det/anydepth-yolov12
D=trt_bench/onnx/bdd_pooled
P=fig8_jetson/assets/policy_both_0.pt
for beta in 0.0 0.3 0.6; do
  echo "=== beta $beta ==="
  python -m method02_advantage_regress_tinyConv.online_budget_demo_stream \
    --base "$D/base.fp16.engine" --super "$D/super.fp16.engine" \
    --router_engine "$D/router.fp16.engine" --policy "$P" \
    --scenarios fig8_jetson/assets/scenarios.json --mot_root fig8_jetson/bdd100k_mot/val \
    --imgsz 720 1280 --grid 2 --win 60 --kp 0.16 --ki 0.08 --beta "$beta" \
    --families night_dawn \
    --dump /tmp/lb$beta.json --out /tmp/lb$beta.pdf 2>&1 | grep -E "night_dawn "
done
echo "=== LOWBETA SWEEP DONE ==="
