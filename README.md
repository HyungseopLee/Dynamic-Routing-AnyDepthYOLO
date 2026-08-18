# Advantage Regression for Adaptive Autonomous Driving Detectors

A lightweight router predicts the per-frame **advantage** of running the deep path over the shallow path of a frozen [AnyDepth-YOLO](https://github.com/AnyDepth/AnyDepth-YOLO) detector. Routing reduces to a single threshold test on the predicted advantage — sweep the threshold to trade accuracy for efficiency without retraining.

---

## Results

| | KITTI | BDD100K | Waymo |
|---|---|---|---|
| **Latency saved** | −12.9% | −17.9% | −20.5% |
| **Energy saved** | −13.0% | −18.2% | −20.5% |
| **AP drop** | < 0.1 | < 0.1 | < 0.1 |

<p align="center">
  <img src="method_advantage_regress/outputs/figures/fig_main_kitti_ap5095.png" width="32%"/>
  <img src="method_advantage_regress/outputs/figures/fig_main_bdd_ap5095.png" width="32%"/>
  <img src="method_advantage_regress/outputs/figures/fig_main_waymo_ap5095.png" width="32%"/>
</p>
<p align="center"><em>(a) KITTI &nbsp;&nbsp;&nbsp;&nbsp; (b) BDD100K &nbsp;&nbsp;&nbsp;&nbsp; (c) Waymo</em></p>

---

## Router

```
method_advantage_regress/
├── router/
│   ├── policy_net.py   # TinyConv-MLP router + GAP-MLP variant
│   ├── feature_tap.py  # Which detector layers to tap (backbone 4,6,8 + neck 14,17,20)
│   └── loss.py         # Advantage regression loss
├── train/
│   ├── build_cache.py  # Step 1: precompute features & losses offline
│   └── train_policy.py # Step 2: train router
└── eval/
    ├── eval_video_bdd.py      # Step 3: BDD100K MOT
    ├── eval_video.py          # Step 3: KITTI tracking
    ├── eval_video_waymo.py    # Step 3: Waymo
    └── merge_video_shards.py
```

---

## Usage

All commands run from the repo root (`anydepth-yolov12/`) with `conda activate yolov12`.

### Step 0 — Finetune AnyDepth-YOLO

The router depends on a frozen detector. Download pretrained COCO weights or finetune from scratch.

**COCO pretrained weights:** [AnyDepth-S (30 MB)](https://drive.google.com/open?id=1XjNjA20ttI54EkFLpys4jSd5fjJkNB5w) · [AnyDepth-L (106 MB)](https://drive.google.com/open?id=1XjNjA20ttI54EkFLpys4jSd5fjJkNB5w)
→ place in `pretraining_AnyDepthYOLO/weights/`

```bash
python -m torch.distributed.run --nproc_per_node 2 finetuning_AnyDepthYOLO/finetune.py \
    --config ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
    --weight pretraining_AnyDepthYOLO/weights/yolo-ad-v12-orig_small_0.479_0.451.pt \
    --data ultralytics/cfg/datasets/<dataset>.yaml --dataset <dataset> \
    --imgsz <imgsz>
```

| Dataset | `--dataset` | `--imgsz` | `--weight` | epochs |
|---------|------------|-----------|------------|--------|
| KITTI | `kitti` | `384 1248` | COCO pretrained | 20 |
| BDD100K | `bdd100k` | `720 1280` | COCO pretrained | 30 |
| Waymo | `waymo` | `1280 1920` | BDD100K finetuned¹ | 30 |

> ¹ Waymo is initialized from the BDD100K finetuned checkpoint (not COCO) because BDD100K provides a better backbone initialization for Waymo's driving domain. The detection head is automatically re-initialized for Waymo's 4-class taxonomy.

**Or download finetuned weights directly:**
[KITTI (15 MB)](https://drive.google.com/open?id=1LZItn0HXmDF-NipuszUVro5HsOu38gEg) · [BDD100K (16 MB)](https://drive.google.com/open?id=1-l01lUpAIGVzYZuw8yNb8yLOWp3_u_sl) · [Waymo (16 MB)](https://drive.google.com/open?id=14WlWrABEcpspwbeecfGlM4pioXPqazfM)
→ place in `finetuning_AnyDepthYOLO/weights/{kitti,bdd100k,waymo}/`

---

### Step 1 — Offline caching

Run the frozen detector once per split (train + val) to precompute features and per-image losses. All datasets use `--feat both --grid 2`.

```bash
python -m method_advantage_regress.train.build_cache \
    --weight <weight> --data <data.yaml> --dataset <dataset> \
    --split <train|val> --imgsz <imgsz> --feat both --grid 2 --batch 16
```

| Dataset | `--weight` | `--data` | `--imgsz` | notes |
|---------|-----------|----------|-----------|-------|
| KITTI | `finetuning_AnyDepthYOLO/weights/kitti/best.pt` | `kitti.yaml` | `384 1248` | |
| BDD100K | `finetuning_AnyDepthYOLO/weights/bdd100k/best.pt` | `bdd100k.yaml` | `720 1280` | |
| Waymo | `finetuning_AnyDepthYOLO/weights/waymo/best.pt` | `waymo.yaml` | `1280 1920` | add `--fp16` |

**Or download prebuilt caches** (feat=both, grid=2):
[KITTI (323 MB)](https://drive.google.com/open?id=1KGSkYIsteuygqxJ29yNW3qOwKGnW_-mh) · [BDD100K (3.4 GB)](https://drive.google.com/open?id=1zPmAZZ2g2Lp6FatyWLRv1tdIM7d5gdNV) · [Waymo (8.4 GB)](https://drive.google.com/open?id=1LpD31l4pj2iE_0LfefF0a_LpGHRtxRNa)
→ place in `method_advantage_regress/outputs/{kitti,bdd100k,waymo}/`

---

### Step 2 — Train the router

```bash
python -m method_advantage_regress.train.train_policy \
    --dataset <dataset> --feat both --norm batch \
    --cache method_advantage_regress/outputs/<dataset>/cache_train_both.pt \
    --val_cache method_advantage_regress/outputs/<dataset>/cache_val_both.pt \
    --select val_corr --mode regress --seed <0-4> \
    --epochs <epochs> \
    --out method_advantage_regress/outputs/<dataset>/policy_s<seed>.pt
```

| Dataset | `--epochs` |
|---------|-----------|
| KITTI | 50 |
| BDD100K | 30 |
| Waymo | 50 |

Repeat for seeds 0–4. Use `--arch gapmpl` for the GAP-MLP variant.

---

### Step 3 — Video evaluation

```bash
python -m method_advantage_regress.eval.<eval_script> \
    --weight <weight> \
    --policies "s0=.../policy_s0.pt,s1=.../policy_s1.pt,..." \
    --val_cache method_advantage_regress/outputs/<dataset>/cache_val_both.pt \
    --grid 2 --imgsz <imgsz> --conf 0.25 \
    --budgets 10,20,30,40,50,60,70,80,90 \
    --raw_out method_advantage_regress/outputs/<dataset>/eval/shard_0.pt
```

| Dataset | `<eval_script>` | `--weight` | `--imgsz` |
|---------|----------------|-----------|-----------|
| KITTI | `eval_video` | `finetuning_AnyDepthYOLO/weights/kitti/best.pt` | `384 1248` |
| BDD100K | `eval_video_bdd` | `finetuning_AnyDepthYOLO/weights/bdd100k/best.pt` | `720 1280` |
| Waymo | — | — | — |

> Waymo uses 4 parallel shards across 2 GPUs: `bash method_advantage_regress/eval/run_waymo_eval_both_robust.sh`

Merge shards into a curve JSON:
```bash
python -m method_advantage_regress.eval.merge_video_shards \
    --shards method_advantage_regress/outputs/<dataset>/eval/shard_0.pt \
    --out method_advantage_regress/outputs/<dataset>/eval/video_curve.json
```
