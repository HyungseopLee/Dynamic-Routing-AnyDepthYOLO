#!/usr/bin/env bash
# Measure TensorRT inference latency / FPS / energy of the AnyDepth detector's two
# depth paths on the Jetson Orin Nano, per dataset, and save the results to:
#   results/step4_deploy/{kitti,bdd100k,waymo}.log
#
# For each dataset it builds the two resident FP16 engines (BASE = all skippable
# layers skipped, SUPER = none) plus the TRT router, then CUDA-event-times:
#   BASE, SUPER, ALTERNATING (switch every frame -> switching overhead), and
#   detector+router combined (the two anchors the main table interpolates between).
#
# Run from the repo root.  MAXN + locked clocks REQUIRED:
#     sudo nvpmodel -m 0 && sudo jetson_clocks
set -euo pipefail

JB=step4_deploy
ENG=$JB/onnx

# dataset | H W | log name | detector weight | router
run() {
  local ds="$1" H="$2" W="$3" log="$4" weight="$5" router="$6"
  local out="$ENG/${ds}_pooled"

  # 1) detector -> base.onnx + super.onnx (2x2 pool baked in) -> FP16 engines
  python $JB/export_onnx.py --weight "$weight" --imgsz "$H" "$W" --out_dir "$out" --pool --grid 2
  python $JB/build_engine.py --onnx "$out/base.onnx"  --fp16
  python $JB/build_engine.py --onnx "$out/super.onnx" --fp16

  # 2) router -> router.onnx -> FP16 engine
  python $JB/export_router_onnx.py --router "$router" --base_engine "$out/base.fp16.engine" --out "$out/router.onnx"
  python $JB/build_engine.py --onnx "$out/router.onnx" --fp16

  # 3) benchmark -> log (BASE / SUPER / ALTERNATING + switch overhead + detector+router)
  python $JB/bench_trt_jetson.py \
      --base          "$out/base.fp16.engine" \
      --super         "$out/super.fp16.engine" \
      --router_engine "$out/router.fp16.engine" \
      --iters 1000 --warmup 100 \
      | tee "$JB/outputs/${log}"
}

run kitti  384 1248  kitti.log    results/step1_finetune/weights/kitti/best.pt \
    results/step3_eval/ablation/kitti/router_both_g2_s0.pt

run bdd    736 1280  bdd100k.log  results/step1_finetune/weights/bdd100k/best.pt \
    results/step2_router/weights/bdd100k/router_both_0.pt

run waymo 1280 1920  waymo.log    results/step1_finetune/weights/waymo/best.pt \
    results/step2_router/weights/waymo/router_both_0.pt
