# Dynamic Depth Routing for Budget-Adaptive Object Detection in ADAS

A lightweight router predicts, per frame, the **advantage** of running the deep (SUPER) path
over the shallow (BASE) path of a frozen [AnyDepth-YOLO](https://github.com/AnyDepth/AnyDepth-YOLO)
detector. Routing then reduces to a single threshold test on that prediction — sweep the
threshold to trade accuracy for efficiency **without retraining**. A feedforward + PI
controller turns the threshold into a live knob that tracks a time-varying latency / FPS /
energy budget on-device.

> **Paper:** *Dynamic Depth Routing for Budget-Adaptive Object Detection in ADAS*, IEEE Access.
> PDF: [`results/paper/`](results/paper/)

| | KITTI | BDD100K | Waymo |
|---|---|---|---|
| **Latency saved** | −12.9% | −17.9% | −20.5% |
| **Energy saved** | −13.0% | −18.2% | −20.5% |
| **AP drop** | < 0.1 | < 0.1 | < 0.1 |

Budget tracking on Jetson Orin Nano (TensorRT FP16, BDD100K 720×1280) holds step and
sawtooth targets to **< 1 ms MAE**.

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Edit `ultralytics/cfg/datasets/{kitti,bdd100k,waymo}.yaml` to point at your dataset roots.
All commands run from the repo root.

### Pretrained weights

Weights and TensorRT engines are distributed via Google Drive (too large for git).
Download and place them as shown:

| Download | Destination |
|---|---|
| Finetuned detector `best.pt` | `results/step1_finetune/weights/<dataset>/` |
| Router `router_both_0.pt` | `results/step2_router/weights/<dataset>/` |
| TensorRT engines (BDD100K) | `results/step4_deploy/onnx/bdd_pooled/` |

With these you can skip straight to [Step 3](#step-3--evaluate) or [Step 4](#step-4--deploy).
Steps 1–2 are only needed to retrain from scratch.

---

The pipeline has four stages, and `results/` mirrors them. BDD100K is used as the running
example below; substitute `<dataset>` and `<imgsz>` from this table for the others.

| Dataset | `<imgsz>` | epochs (detector / router) |
|---|---|---|
| KITTI | `384 1248` | 20 / 50 |
| BDD100K | `720 1280` | 30 / 30 |
| Waymo | `1280 1920` | 30 / 50 |

## Step 1 — Finetune the detector

The router needs a **frozen** detector, so this comes first.

```bash
python -m torch.distributed.run --nproc_per_node 2 step1_finetune/finetune.py \
    --config ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
    --weight results/step1_finetune/pretrained_coco/yolo-ad-v12-orig_small_0.479_0.451.pt \
    --data ultralytics/cfg/datasets/bdd100k.yaml --dataset bdd100k \
    --imgsz 720 1280
```

→ `results/step1_finetune/weights/bdd100k/best.pt`

> Waymo starts from the BDD100K checkpoint instead of COCO (closer driving domain); its
> head is re-initialized for the 4-class taxonomy.

## Step 2 — Train the router

Cache the router features and both paths' per-image losses once (their difference is the
regression target), then fit the router on that cache.

```bash
# 2a. offline cache (repeat with --split val; Waymo also needs --fp16)
python -m step2_train_router.build_cache \
    --weight results/step1_finetune/weights/bdd100k/best.pt \
    --data ultralytics/cfg/datasets/bdd100k.yaml --dataset bdd100k \
    --split train --imgsz 720 1280 --feat both --grid 2 --batch 16

# 2b. advantage regression
python -m step2_train_router.train_policy --dataset bdd100k \
    --cache     results/step2_router/cache/bdd100k/cache_train_both.pt \
    --val_cache results/step2_router/cache/bdd100k/cache_val_both.pt \
    --feat both --norm batch --select val_corr --mode regress \
    --epochs 30 --seed 0 \
    --out results/step2_router/weights/bdd100k/router_both_0.pt
```

→ `results/step2_router/weights/bdd100k/router_both_0.pt`

## Step 3 — Evaluate

Routing is evaluated **causally**: the advantage predicted at frame *t−1* selects the path
run at frame *t*, so no future information leaks in.

```bash
python -m step3_eval.eval_video --dataset bdd100k \
    --weight    results/step1_finetune/weights/bdd100k/best.pt \
    --policies  "s0=results/step2_router/weights/bdd100k/router_both_0.pt" \
    --val_cache results/step2_router/cache/bdd100k/cache_val_both.pt \
    --grid 2 --imgsz 720 1280 --conf 0.25 \
    --budgets 10,20,30,40,50,60,70,80,90 \
    --raw_out results/step3_eval/bdd100k/eval/shard_0.pt

python -m step3_eval.merge_video_shards \
    --shards results/step3_eval/bdd100k/eval/shard_0.pt \
    --out    results/step3_eval/bdd100k/eval/video_curve.json
```

The paper reports the mean over seeds 0–4 — pass them as a comma-separated `--policies`
list to reproduce that. Ablations live in [`step3_eval/ablation/`](step3_eval/ablation/).

## Step 4 — Deploy

TensorRT cannot skip layers inside one static engine, so each depth path is exported as its
own engine and the router picks which one to run per frame.

```bash
# 4a. export + build (--pool bakes the 2x2 tap pooling in: ~2 ms/frame on Jetson)
python -m step4_deploy.export_onnx \
    --weight results/step1_finetune/weights/bdd100k/best.pt \
    --imgsz 720 1280 --pool \
    --out_dir results/step4_deploy/onnx/bdd_pooled

python -m step4_deploy.export_router_onnx \
    --router results/step2_router/weights/bdd100k/router_both_0.pt \
    --out_dir results/step4_deploy/onnx/bdd_pooled

for m in base super router; do
    python -m step4_deploy.build_engine \
        --onnx results/step4_deploy/onnx/bdd_pooled/$m.onnx --fp16
done
```

```bash
# 4b. track a time-varying budget with the PI controller
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
Results land in `results/step4_deploy/control/`. Controller gains, per device:

| Device | Resolution | Gains |
|---|---|---|
| Jetson Orin Nano | 576×1024 | `--kp 1.1 --ki 0.18 --beta 0.95 --win 60` |
| Jetson Orin Nano | 720×1280 | `--kp 2.0 --ki 0.33 --beta 0.85 --win 30` |
| RTX 3090 | 720×1280 | `--kp 1.0 --ki 0.10 --beta 0.85 --win 60` |

```bash
# 4c. demo video: detections + BASE/SUPER badge + realized-vs-target FPS
python -m step4_deploy.online_demo_video \
    --base   results/step4_deploy/onnx/bdd_pooled/base.fp16.engine \
    --super  results/step4_deploy/onnx/bdd_pooled/super.fp16.engine \
    --router results/step2_router/weights/bdd100k/router_both_0.pt \
    --router_engine results/step4_deploy/onnx/bdd_pooled/router.fp16.engine \
    --scenarios results/step3_eval/bdd100k/scenarios.json \
    --mot_root  /media/data/bdd100k_mot/val \
    --kp 1.0 --ki 0.10 --beta 0.85
```

→ `results/step4_deploy/demo_videos/demo_video_<family>_fps<lo>-<hi>.mp4`

For the true deployed cost of routing (only one path runs per frame), use
`step4_deploy.trt_video_eval`.

---

## Layout

```
step1_finetune/  step2_train_router/  step3_eval/  step4_deploy/   pipeline stages
router/          router architecture, feature taps, loss
analysis/        paper figures and tables  ->  results/figures/
tools/           dataset conversion (KITTI / BDD100K / Waymo -> YOLO)
ultralytics/     modified YOLOv12 backbone with skip-layer depth control
results/         mirrors the four stages, plus figures/ and paper/
```

Each script's module docstring carries a runnable `Usage` example.

## License

[AGPL-3.0](LICENSE), inherited from Ultralytics.
