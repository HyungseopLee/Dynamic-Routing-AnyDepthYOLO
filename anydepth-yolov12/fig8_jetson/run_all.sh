#!/usr/bin/env bash
# Autonomous Fig-8 (BDD) TensorRT pipeline. Safe to leave running unattended:
#   1. wait for the base/super pooled engine build to finish
#   2. export + build the single TRT router engine (iv,pv->logit, pid baked)
#   3. run the memory-safe streaming budget-tracking demo (detector + router in TRT)
#   4. render the figure
# All steps logged to fig8_jetson/*.log; the streaming demo holds one frame at a time
# (no OOM/reboot risk). Clocks are assumed already locked (sudo jetson_clocks, 1020MHz).
set -e
cd /home/hslee/context-anydepth-det/anydepth-yolov12
D=trt_bench/onnx/bdd_pooled
P=fig8_jetson/assets/policy_both_0.pt

echo "=== [1/4] waiting for base/super engines ==="
# wait until the prior build process drops both engine files
while [ ! -f "$D/base.fp16.engine" ] || [ ! -f "$D/super.fp16.engine" ]; do
  sleep 15
done
sleep 5
echo "base/super engines present:"; ls -la "$D"/*.engine

echo "=== [2/4] export + build TRT router engine ==="
python trt_bench/export_router_onnx.py \
  --policy "$P" --base_engine "$D/base.fp16.engine" \
  --out "$D/router.onnx" --pid 1
python trt_bench/build_engine.py --onnx "$D/router.onnx" --fp16 --workspace_gb 4
ls -la "$D/router.fp16.engine"

echo "=== [3/4] gpu clock check ==="
cat /sys/class/devfreq/17000000.gpu/cur_freq

echo "=== [3/4] run streaming TRT budget-tracking demo ==="
python -m method02_advantage_regress_tinyConv.online_budget_demo_stream \
  --base  "$D/base.fp16.engine" \
  --super "$D/super.fp16.engine" \
  --router_engine "$D/router.fp16.engine" \
  --policy "$P" \
  --scenarios fig8_jetson/assets/scenarios.json \
  --mot_root  fig8_jetson/bdd100k_mot/val \
  --imgsz 720 1280 --grid 2 --win 60 --kp 0.040 --ki 0.020 \
  --dump fig8_jetson/online_budget_demo_jetson.json \
  --out  fig8_jetson/fig8_jetson.pdf

echo "=== [4/4] ALL DONE ==="
ls -la fig8_jetson/fig8_jetson.pdf fig8_jetson/online_budget_demo_jetson.json
