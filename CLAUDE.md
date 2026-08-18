# Project: Dynamic Depth Routing for Budget-Adaptive Object Detection in ADAS

Conda env: `yolov12`. All commands run from repo root without `conda run` prefix.

## Repository structure

Code is organized by pipeline stage; `results/` mirrors the same four stages.

```
step1_finetune/        finetune.py, pretrain_coco.py
step2_train_router/    build_cache.py, train_policy.py
step3_eval/            eval_video.py (unified KITTI/BDD/Waymo), eval_utils.py,
                       eval_baseline_kitti.py, merge_video_shards.py, ablation/
step4_deploy/          export_onnx.py, export_router_onnx.py, build_engine.py,
                       online_budget_demo_stream.py, online_demo_video.py,
                       trt_video_eval.py, bench/
router/                router_net.py (TinyConv + GapMlpNet), feature_tap.py, loss.py
analysis/              make_main_figure.py, make_calibration.py, plot_*.py
tools/                 dataset conversion
ultralytics/           modified YOLO backbone (do not touch)
paper/                 manuscript PDF + source figures

results/
├── step1_finetune/    pretrained_coco/, weights/<ds>/best.pt, logs/
├── step2_router/      cache/<ds>/cache_*.pt, weights/<ds>/router_*.pt
├── step3_eval/        <ds>/eval/, <ds>/scenarios.json, ablation/
├── step4_deploy/      onnx/<engdir>/, control/, demo_videos/
└── figures/
```

Total on-disk `results/` is ~55 GB. `.gitignore` excludes `*.pt`, `*.pkl`, `*.onnx`,
`*.engine`, `*.mp4`, the cache tree, and step1 logs — tracked content is ~65 MB.
Large artifacts are distributed via Google Drive instead.

## eval_baseline_kitti

Provides AP matching and mAP computation. It was previously shipped only as a
`.pyc`; the source was restored from git history (commit `28e0286`) and now lives at
`step3_eval/eval_baseline_kitti.py`. `step3_eval/eval_utils.py` is a thin shim that
re-exports it and registers it in `sys.modules` under its bare name.

## Key design decisions

### eval_video.py — unified eval (KITTI/BDD100K/Waymo)
`--dataset kitti/bdd100k/waymo` selects data loading. Public aliases for backward compat:
- `parse_box_track`, `labeled_frames` (BDD helpers)
- `BDD_MOT_EVAL_CLS`

### Router loading
Auto-detects TinyConv vs GapMlpNet by checkpoint weight dimensionality:
```python
is_gap = not any(k.endswith("weight") and v.dim() == 4 for k, v in sd.items())
```
Checkpoint key: `state_dict` (not `model_state`).
`policy_both_0.pt` is **feat=both** → must pass both `xi` and `xp` to `net.logit(xi, xp, pid)`.

### Feature taps
- `INPUT_LEVEL_LAYERS = [4, 6, 8]` (backbone)
- `PRED_LEVEL_LAYERS = [14, 17, 20]` (neck)

## TRT pipeline (BDD100K)

Engines: `results/step4_deploy/onnx/bdd_pooled/{base,super,router}.fp16.engine`

PI controller tuning — `gscale = (TAU_HI - TAU_LO) / (l_super - l_base)` (auto-computed):

- **RTX 3090 TRT (720×1280 BDD), FPS mode:** `--kp 1.0 --ki 0.10 --beta 0.85 --win 60`
  - l_base=3.24ms, l_super=5.20ms; FPS range ~199–292
  - Step MAE 0.115 ms, sawtooth MAE 0.146 ms (best of the sweep)
  - Note: `--kp 0.28 --ki 0.06 --beta 0.93` was tuned for *latency* mode and is a poor
    fit for FPS mode (MAE ~8–14 fps).
- **Jetson Orin Nano TRT (576×1024 BDD):** `--kp 1.1 --ki 0.18 --beta 0.95 --warmup 60 --win 60`
  - l_base=14.45ms, l_super=23.37ms
  - MAE: step 0.30~0.37 ms, sawtooth 0.66~0.73 ms (all < 1 ms)
  - Code changes in `online_budget_demo_stream.py`: per-family τ init + L_ema init at Ltgt[0]
- **Jetson Orin Nano TRT (720×1280 BDD, original res):** `--kp 2.0 --ki 0.33 --beta 0.85 --warmup 60 --win 30`
  - l_base=25.12ms, l_super=42.12ms
  - MAE: step 0.37~0.44 ms, sawtooth 0.55~0.60 ms (all < 1 ms)
- PyTorch backend params: `--kp 0.020 --ki 0.004 --beta 0.85`

Controller sweep outputs (JSON + PDF) live in `results/step4_deploy/control/`.

## Demo video

`step4_deploy/online_demo_video.py` — uses `all_frames()` (every video frame, ~6000 per
family at 30 fps) rather than `labeled_frames()` (annotation frames only, ~1000 at 5 fps).
The measured/target FPS overlay refreshes once per second (`frame_idx % round(out_fps)`)
so the numbers stay readable. Final videos: `demo_video_{night_dawn,city_highway,clear_rainy}_fps199-292.mp4`.

## Google Drive

rclone remote: `gdrive`, folder: `AnyDepth-Router/`
Watchdog script: `~/watchdog_upload.sh`

## Paper main results

| Dataset | Latency saved | Energy saved | AP drop |
|---------|--------------|--------------|---------|
| KITTI   | −12.9%       | −13.0%       | < 0.1   |
| BDD100K | −17.9%       | −18.2%       | < 0.1   |
| Waymo   | −20.5%       | −20.5%       | < 0.1   |
