# method02 — tinyConv advantage-regression depth router (MAIN METHOD)

A lightweight **router** decides, per video frame, whether the **frozen**
AnyDepth-YOLOv12s detector runs at **BASE** (16.87 GFLOPs) or **SUPER** (26.30 GFLOPs)
depth, minimizing compute while keeping AP. The router regresses the per-frame
**advantage** `A = L_base − L_super` (MSE); at deploy a single threshold `Â > τ → SUPER`
sweeps the whole compute/accuracy curve (no retraining per budget).

Router input = backbone layers {4,6,8} adaptive-pooled to a **2×2 grid**, then a tiny
spatial conv (1×1 C→d → ReLU → 3×3 depthwise → ReLU → GAP) → MLP → `Â`.
(method01 = the GAP-MLP variant that collapses spatial layout; this method beats it.)

## Environment
- Detector forward / cache / training: base conda env (torch + ultralytics).
- **`eval_video.py` needs `conda run -n yolov12`** (torchvision required by the
  `yolo.predict` GPU warmup). All other steps run in the base env.
- All commands are run from the repo root `anydepth-yolov12/`.

## Experimental setup (defaults)
| | value |
|---|---|
| Detector weight | `runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt` (frozen) |
| Depth levels | BASE = skip[ ]·8 = True (16.87 G) / SUPER = False·8 (26.30 G) |
| Train data | KITTI-detection cache: 5985 train / 1496 val (`ultralytics/cfg/datasets/kitti.yaml`) |
| Eval data | KITTI-tracking video, 21 seq / 8008 frames (`/media/data/kitti-tracking`) |
| Image size | 384 × 1248 |
| conf | **0.25** (all strategies share the same conf) |
| Router | feat=backbone, grid=2, group_dim=64, hidden=128, path_dim=8, norm=batch |
| Loss / select | MSE(`Â`,`A`) / **val-corr** checkpoint selection |
| Seeds | 5 (report mean ± std) |

---

## Pipeline

### 0. (one-time) FLOPs lookup table
Per-action GFLOPs (BASE/SUPER), used by the loss and by eval normalization.
```bash
python method02_advantage_regress_tinyConv/build_flops_table.py \
  --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
  --imgsz 384 1248 --dataset kitti
# -> outputs/kitti/flops_table.json
```

### 1. Offline caching
Runs the frozen detector over KITTI-detection once, storing per-image pooled feature
grids (both BASE & SUPER paths) + per-image losses. Training then reads tensors only.
`--grid` = `G` (square G×G) or `HxW` (rectangular, aspect-preserving).
```bash
for split in train val; do
  python method02_advantage_regress_tinyConv/build_cache.py \
    --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
    --data ultralytics/cfg/datasets/kitti.yaml --dataset kitti \
    --split $split --imgsz 384 1248 --batch 16 --grid 2 \
    --out method02_advantage_regress_tinyConv/outputs/kitti/cache_${split}_g2.pt
done
```

### 2. Training (5 seeds)
Cheap (reads cache, no detector forward). One model traces the full curve.
```bash
O=method02_advantage_regress_tinyConv/outputs/kitti
for s in 0 1 2 3 4; do
  python method02_advantage_regress_tinyConv/train_policy.py \
    --dataset kitti --cache $O/cache_train_g2.pt --val_cache $O/cache_val_g2.pt \
    --flops $O/flops_table.json --epochs 300 --batch 256 --lr 0.001 \
    --hidden 128 --group_dim 64 --path_dim 8 --feat input --norm batch \
    --select val_corr --mode regress --regress_loss mse --prev_p 0.5 --seed $s \
    --out $O/ablation/policy_input_g2_s$s.pt --logdir $O/ablation/logs --tag g2_s$s
done
```
Useful axes: `--grid` (cache must match), `--keep_layers 4|6|8` (single backbone level,
channel-slice, no recache), `--feat input|pred|both`, `--path_dim 0` (no prev-action),
`--regress_loss mse|mae|huber|corr`, `--select val_corr|val_mse|last`.

### 3. Evaluation (video, full τ sweep)  ⚠ needs `yolov12` env
Recursive temporal routing: frame *t*'s path is decided from frame *t−1*'s chosen-path
feature. `--policy_only` skips the lum/edge/conf/random baselines. latency/FPS/energy are
measured on the stream **including the router forward**.
```bash
A=method02_advantage_regress_tinyConv/outputs/kitti/ablation
E=method02_advantage_regress_tinyConv/outputs/kitti/eval
POL="input_s0=$A/policy_input_g2_s0.pt,...,input_s4=$A/policy_input_g2_s4.pt"
conda run -n yolov12 python method02_advantage_regress_tinyConv/eval_video.py \
  --policies "$POL" --grid 2 --conf 0.25 --policy_only \
  --out $E/video_curve_tinyconv_g2.json
```
Drop `--policy_only` (and keep all baselines) to produce the main-result curve.

### 4. Plots & tables
```bash
# multi-curve compare (ablations): label:path,label:path
python method01_advantage_regress/plot_norm_compare.py \
  --curves "conv-2x2:$E/video_curve_tinyconv_g2.json,GAP-MLP:<m01>/video_curve_selcorr.json" \
  --feats input --band --metric map50 --out $E/fig_conv_vs_gapmlp_g2
# main result (policy vs baselines)
python method01_advantage_regress/plot_ablation.py \
  --curve $E/video_curve_main_g2_merged.json --with_conf --feats input \
  --metric map50 --out $E/fig_main_g2
```

### 5. (optional) Honest threshold pipeline
Fix τ on val per FLOPs budget, apply only on video:
```bash
python method02_advantage_regress_tinyConv/get_thresholds.py \
  --dataset kitti --policy_glob "$A/policy_input_g2_s*.pt" \
  --budgets 10,20,30,40,50,60,70,80,90 --name val_thresholds_tinyconv_backbone
# then eval_video with --val_taus_json <that json>
```

### 6. Router overhead
```bash
python method02_advantage_regress_tinyConv/measure_router_overhead.py --grid 2 --group_dim 64 --hidden 128
# tinyConv 2×2: 60,881 params, 4.17e-4 GFLOPs (0.0015 % of SUPER)
```

## Results
See `outputs/kitti/eval/RESULTS.md` (headline table, all ablations, file pointers) and
`outputs/kitti/eval/tinyconv_g2_backbone_table.md` (full τ sweep with reductions).
