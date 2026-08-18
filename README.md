# Dynamic Depth Routing for Budget-Adaptive Object Detection in ADAS

A lightweight router predicts the per-frame **advantage** of running the deep (SUPER) path
over the shallow (BASE) path of a frozen [AnyDepth-YOLO](https://github.com/AnyDepth/AnyDepth-YOLO)
detector. Routing reduces to a single threshold test on the predicted advantage — sweep the
threshold to trade accuracy for efficiency **without retraining**. A feedforward + PI
controller then turns that threshold into a live knob that tracks a time-varying
latency / FPS / energy budget on-device.

> **Paper:** *Dynamic Depth Routing for Budget-Adaptive Object Detection in ADAS*, IEEE Access.
> PDF in [`paper/`](paper/).

---

## Results

| | KITTI | BDD100K | Waymo |
|---|---|---|---|
| **Latency saved** | −12.9% | −17.9% | −20.5% |
| **Energy saved** | −13.0% | −18.2% | −20.5% |
| **AP drop** | < 0.1 | < 0.1 | < 0.1 |

Budget tracking on Jetson Orin Nano (TensorRT FP16, BDD100K 720×1280) holds step and
sawtooth targets to **< 1 ms MAE**.

---

## Repository layout

The repo follows the four stages of the pipeline; `results/` mirrors that same structure.

```
step1_finetune/        1. Finetune AnyDepth-YOLO on the target dataset
step2_train_router/    2. Cache features offline, then train the advantage router
step3_eval/            3. Video evaluation with causal routing (+ ablation/)
step4_deploy/          4. ONNX/TensorRT export, PI budget control, demo video

router/                Router architecture (TinyConv / GAP-MLP), feature taps, loss
analysis/              Paper figure + table generation
tools/                 Dataset conversion (KITTI / BDD100K / Waymo -> YOLO)
ultralytics/           Modified YOLOv12 backbone with skip-layer depth control
paper/                 Manuscript PDF and source figures

results/
├── step1_finetune/    pretrained_coco/, weights/{kitti,bdd100k,waymo}/, logs/
├── step2_router/      cache/<dataset>/, weights/<dataset>/
├── step3_eval/        <dataset>/eval/, ablation/
├── step4_deploy/      onnx/, control/, demo_videos/
└── figures/           All paper figures
```

### Downloads

Weights, caches, TensorRT engines, and demo videos are too large for git and are
distributed via Google Drive. Place each archive in the matching `results/` directory.

| Artifact | Destination |
|---|---|
| COCO-pretrained AnyDepth-YOLO | `results/step1_finetune/pretrained_coco/` |
| Finetuned detectors (KITTI / BDD100K / Waymo) | `results/step1_finetune/weights/<dataset>/` |
| Offline caches (feat=both, grid=2) | `results/step2_router/cache/<dataset>/` |
| Trained routers (seeds 0–4) | `results/step2_router/weights/<dataset>/` |
| ONNX + TensorRT engines | `results/step4_deploy/onnx/` |
| Demo videos | `results/step4_deploy/demo_videos/` |

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Then edit `ultralytics/cfg/datasets/{kitti,bdd100k,waymo}.yaml` to point at your dataset roots.
All commands below are run from the repo root.

---

## Step 1 — Finetune AnyDepth-YOLO

The router depends on a **frozen** detector, so this comes first.

```bash
python -m torch.distributed.run --nproc_per_node 2 step1_finetune/finetune.py \
    --config ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
    --weight results/step1_finetune/pretrained_coco/yolo-ad-v12-orig_small_0.479_0.451.pt \
    --data ultralytics/cfg/datasets/<dataset>.yaml --dataset <dataset> \
    --imgsz <imgsz>
```

| Dataset | `--dataset` | `--imgsz` | initial `--weight` | epochs |
|---------|------------|-----------|--------------------|--------|
| KITTI | `kitti` | `384 1248` | COCO pretrained | 20 |
| BDD100K | `bdd100k` | `720 1280` | COCO pretrained | 30 |
| Waymo | `waymo` | `1280 1920` | BDD100K finetuned¹ | 30 |

> ¹ Waymo starts from the BDD100K checkpoint rather than COCO — BDD100K is the closer
> driving domain. The head is re-initialized automatically for Waymo's 4-class taxonomy.

Produces `results/step1_finetune/weights/<dataset>/best.pt`.

---

## Step 2 — Train the router

### 2a. Offline caching

Run the frozen detector once per split to precompute router features and the per-image
loss of both paths (their difference is the regression target). All datasets use
`--feat both --grid 2`.

```bash
python -m step2_train_router.build_cache \
    --weight results/step1_finetune/weights/<dataset>/best.pt \
    --data ultralytics/cfg/datasets/<dataset>.yaml --dataset <dataset> \
    --split <train|val> --imgsz <imgsz> --feat both --grid 2 --batch 16
```

Waymo additionally needs `--fp16`. Produces
`results/step2_router/cache/<dataset>/cache_{train,val}_both.pt`.

### 2b. Advantage regression

```bash
python -m step2_train_router.train_policy \
    --dataset <dataset> --feat both --norm batch \
    --cache     results/step2_router/cache/<dataset>/cache_train_both.pt \
    --val_cache results/step2_router/cache/<dataset>/cache_val_both.pt \
    --select val_corr --mode regress --seed <0-4> --epochs <epochs> \
    --out results/step2_router/weights/<dataset>/router_both_<seed>.pt
```

Epochs: KITTI 50, BDD100K 30, Waymo 50. Repeat for seeds 0–4.
Use `--arch gapmpl` for the GAP-MLP variant.

---

## Step 3 — Video evaluation (causal routing)

Routing is evaluated **causally**: the advantage predicted at frame *t−1* selects the path
executed at frame *t*, so no future information leaks in.

```bash
python -m step3_eval.eval_video --dataset <dataset> \
    --weight    results/step1_finetune/weights/<dataset>/best.pt \
    --policies  "s0=results/step2_router/weights/<dataset>/router_both_0.pt,s1=...,..." \
    --val_cache results/step2_router/cache/<dataset>/cache_val_both.pt \
    --grid 2 --imgsz <imgsz> --conf 0.25 \
    --budgets 10,20,30,40,50,60,70,80,90 \
    --raw_out results/step3_eval/<dataset>/eval/shard_0.pt
```

Waymo runs as 4 shards across 2 GPUs:

```bash
bash step3_eval/run_waymo_eval_both_robust.sh
```

Merge shards into the accuracy/efficiency curve:

```bash
python -m step3_eval.merge_video_shards \
    --shards results/step3_eval/<dataset>/eval/shard_0.pt \
    --out    results/step3_eval/<dataset>/eval/video_curve.json
```

Ablations (router grid, architecture, feature taps, prev-action) live in
[`step3_eval/ablation/`](step3_eval/ablation/).

---

## Step 4 — Deployment

TensorRT cannot skip layers inside one static engine, so each depth path is exported as
its own engine; the router picks which engine to run per frame.

### 4a. Export ONNX and build engines

```bash
# BASE + SUPER paths (--pool bakes the 2x2 tap pooling into the graph: ~2 ms/frame on Jetson)
python -m step4_deploy.export_onnx \
    --weight results/step1_finetune/weights/bdd100k/best.pt \
    --imgsz 720 1280 --pool \
    --out_dir results/step4_deploy/onnx/bdd_pooled

# Router
python -m step4_deploy.export_router_onnx \
    --router results/step2_router/weights/bdd100k/router_both_0.pt \
    --out_dir results/step4_deploy/onnx/bdd_pooled

# FP16 engines (use --workspace_gb 4 on Jetson)
for m in base super router; do
    python -m step4_deploy.build_engine \
        --onnx results/step4_deploy/onnx/bdd_pooled/$m.onnx --fp16
done
```

### 4b. Budget tracking with the PI controller

Sweeps the scenario families (night↔day, city↔highway, clear↔rainy) against step and
sawtooth targets, in latency / FPS / energy mode.

```bash
python -m step4_deploy.online_budget_demo_stream \
    --base   results/step4_deploy/onnx/bdd_pooled/base.fp16.engine \
    --super  results/step4_deploy/onnx/bdd_pooled/super.fp16.engine \
    --router results/step2_router/weights/bdd100k/router_both_0.pt \
    --router_engine results/step4_deploy/onnx/bdd_pooled/router.fp16.engine \
    --scenarios results/step3_eval/bdd100k/scenarios.json \
    --mot_root  /media/data/bdd100k_mot/val \
    --mode fps --kp 1.0 --ki 0.10 --beta 0.75 --warmup 60 --win 30
```

`--mode` is `latency` (ms, default), `fps`, or `energy` (mJ/frame via NVML).
Results land in `results/step4_deploy/control/`.

**Controller gains** (`kp`/`ki` are normalized internally by the BASE↔SUPER latency spread):

| Device | Resolution | Backend | Gains |
|---|---|---|---|
| Jetson Orin Nano | 576×1024 | TRT FP16 | `--kp 1.1 --ki 0.18 --beta 0.95 --win 60` |
| Jetson Orin Nano | 720×1280 | TRT FP16 | `--kp 2.0 --ki 0.33 --beta 0.85 --win 30` |
| RTX 3090 | 720×1280 | TRT FP16 | `--kp 1.0 --ki 0.10 --beta 0.85 --win 60` |

### 4c. End-to-end TensorRT evaluation

Streams real MOT video through the deployed loop on the engines — only one path runs per
frame, so this measures the true deployed cost of routing.

```bash
python -m step4_deploy.trt_video_eval \
    --base   results/step4_deploy/onnx/bdd_pooled/base.fp16.engine \
    --super  results/step4_deploy/onnx/bdd_pooled/super.fp16.engine \
    --router results/step2_router/weights/bdd100k/router_both_0.pt \
    --mot_root /media/data/bdd100k_mot/val --imgsz 720 1280
```

### 4d. Demo video

Renders detections, the live BASE/SUPER badge, and realized-vs-target FPS while the
controller tracks a continuously varying sawtooth target.

```bash
python -m step4_deploy.online_demo_video \
    --base   results/step4_deploy/onnx/bdd_pooled/base.fp16.engine \
    --super  results/step4_deploy/onnx/bdd_pooled/super.fp16.engine \
    --router results/step2_router/weights/bdd100k/router_both_0.pt \
    --router_engine results/step4_deploy/onnx/bdd_pooled/router.fp16.engine \
    --scenarios results/step3_eval/bdd100k/scenarios.json \
    --mot_root  /media/data/bdd100k_mot/val \
    --kp 1.0 --ki 0.10 --beta 0.85
```

Writes `results/step4_deploy/demo_videos/demo_video_<family>_fps<lo>-<hi>.mp4`.

---

## Reproducing the paper figures

```bash
python -m analysis.make_main_figure       # accuracy/efficiency curves
python -m analysis.make_calibration       # router calibration
python -m analysis.plot_advantage_dist    # advantage distribution
python -m analysis.analyze_router_behavior
```

Figures are written to `results/figures/`.

---

## License

[AGPL-3.0](LICENSE), inherited from Ultralytics.
