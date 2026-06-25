# Project: Advantage Regression for Adaptive Autonomous Driving Detectors

Conda env: `yolov12`. All commands run from repo root without `conda run` prefix.

## Repository structure

```
method_advantage_regress/   ← main router contribution
├── router/                 ← policy_net.py (TinyConv + GapMlpNet), feature_tap.py, loss.py
├── train/                  ← build_cache.py, train_policy.py
├── eval/                   ← eval_video.py (unified KITTI/BDD/Waymo), eval_utils.py
├── analysis/               ← make_main_figure.py, analyze_router_behavior.py
├── ablation/               ← run_bdd_*.sh
├── jetson/                 ← TRT export + budget demo scripts
└── outputs/                ← {kitti,bdd100k,waymo}/  +  figures/
finetuning_AnyDepthYOLO/    ← finetune.py + weights/{kitti,bdd100k,waymo}/best.pt
pretraining_AnyDepthYOLO/   ← COCO pretrained weights
ultralytics/                ← modified YOLO backbone (do not touch)
tools/                      ← dataset conversion scripts
```

## Critical: eval_baseline_kitti.pyc

`eval_baseline_kitti.py` was deleted; only `__pycache__/eval_baseline_kitti.cpython-311.pyc` remains.
`method_advantage_regress/eval/eval_utils.py` loads it via importlib at runtime.
The import is **lazy** (inside `main()` in eval_video.py) — TRT/demo scripts that only import
helper functions (parse_box_track, load_policy, etc.) are not affected.

**Do not delete `__pycache__/`** — it will break `eval_video.main()`.

## Key design decisions

### eval_video.py — unified eval (KITTI/BDD100K/Waymo)
`--dataset kitti/bdd100k/waymo` selects data loading. Public aliases for backward compat:
- `parse_box_track`, `labeled_frames` (BDD helpers)
- `BDD_MOT_EVAL_CLS`

### Policy loading
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

Engines: `method_advantage_regress/jetson/onnx/bdd_pooled/{base,super,router}.fp16.engine`

PI controller tuning for RTX3090 TRT backend:
- `gscale = (TAU_HI - TAU_LO) / (l_super - l_base)` ≈ 0.10 (auto-computed)
- **Good params:** `--kp 0.28 --ki 0.06 --beta 0.93 --warmup 60 --win 60`
- PyTorch backend params: `--kp 0.020 --ki 0.004 --beta 0.85`

## Google Drive

rclone remote: `gdrive`, folder: `AnyDepth-Router/`
Watchdog script: `~/watchdog_upload.sh`

## Paper main results

| Dataset | Latency saved | Energy saved | AP drop |
|---------|--------------|--------------|---------|
| KITTI   | −12.9%       | −13.0%       | < 0.1   |
| BDD100K | −17.9%       | −18.2%       | < 0.1   |
| Waymo   | −20.5%       | −20.5%       | < 0.1   |
