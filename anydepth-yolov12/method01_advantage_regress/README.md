# method01 — GAP-MLP advantage-regression depth router (baseline variant)

A lightweight **router** decides, per video frame, whether the **frozen**
AnyDepth-YOLOv12s detector runs at **BASE** (16.87 GFLOPs) or **SUPER** (26.30 GFLOPs)
depth. The router regresses the per-frame **advantage** `A = L_base − L_super` (MSE);
at deploy a single threshold `Â > τ → SUPER` sweeps the whole compute/accuracy curve.

Router input = backbone layers {4,6,8} → **GAP** → MLP → `Â`. This is the simplest
router; the spatial-conv variant **method02 (tinyConv 2×2) is the main method** and
Pareto-dominates this one. Pipeline is identical except method02 adds a `--grid` axis.

## Environment
- Cache / training: base conda env (torch + ultralytics).
- **`eval_video.py` needs `conda run -n yolov12`** (torchvision for the `yolo.predict`
  warmup). Other steps run in base. Run all commands from repo root `anydepth-yolov12/`.

## Experimental setup (defaults)
| | value |
|---|---|
| Detector weight | `runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt` (frozen) |
| Depth levels | BASE 16.87 G / SUPER 26.30 G |
| Train data | KITTI-detection cache: 5985 train / 1496 val |
| Eval data | KITTI-tracking video, 21 seq / 8008 frames (`/media/data/kitti-tracking`) |
| Image size / conf | 384 × 1248 / **0.25** |
| Router | feat=backbone, group_dim=64, hidden=128, path_dim=8, norm=batch |
| Loss / select / seeds | MSE(`Â`,`A`) / **val-corr** / 5 |

---

## Pipeline

### 0. FLOPs lookup table (one-time)
```bash
python method01_advantage_regress/build_flops_table.py \
  --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
  --imgsz 384 1248 --dataset kitti        # -> outputs/kitti/flops_table.json
```

### 1. Offline caching (frozen detector → per-image GAP features + losses)
```bash
for split in train val; do
  python method01_advantage_regress/build_cache.py \
    --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
    --data ultralytics/cfg/datasets/kitti.yaml --dataset kitti \
    --split $split --imgsz 384 1248 --batch 16 \
    --out method01_advantage_regress/outputs/kitti/cache_${split}.pt
done
```

### 2. Training (5 seeds)
```bash
O=method01_advantage_regress/outputs/kitti
for s in 0 1 2 3 4; do
  python method01_advantage_regress/train_policy.py \
    --dataset kitti --cache $O/cache_train.pt --val_cache $O/cache_val.pt \
    --flops $O/flops_table.json --epochs 300 --batch 256 --lr 0.001 \
    --hidden 128 --group_dim 64 --path_dim 8 --feat input --norm batch \
    --select val_corr --mode regress --regress_loss mse --prev_p 0.5 --seed $s \
    --out $O/sel_corr/policy_input_s$s.pt --logdir $O/sel_corr/logs --tag s$s
done
```
Ablation axes: `--feat input|pred|both`, `--path_dim 0|8|64|...`, `--regress_loss
mse|mae|huber|corr`, `--select val_corr|val_mse|last`, `--norm none|batch|layer`,
`--prev_p 0|0.25|0.5|0.75|1.0`.

### 3. Evaluation (video, full τ sweep)  ⚠ needs `yolov12` env
```bash
E=method01_advantage_regress/outputs/kitti/eval
POL="input_s0=$O/sel_corr/policy_input_s0.pt,...,input_s4=$O/sel_corr/policy_input_s4.pt"
conda run -n yolov12 python method01_advantage_regress/eval_video.py \
  --policies "$POL" --conf 0.25 --policy_only \
  --out $E/video_curve_selcorr.json
```
Drop `--policy_only` to also evaluate lum/edge/conf/random baselines (main-result curve).

### 4. Plots & tables
```bash
# feature ablation (backbone/neck/both) only
python method01_advantage_regress/plot_norm_compare.py \
  --curves "GAP-MLP:$E/video_curve_featabl_seeds.json" --feats input,pred,both \
  --band --metric map50 --out $E/fig_feat_only
# main result (policy vs baselines)
python method01_advantage_regress/plot_ablation.py \
  --curve $E/video_curve_featabl_seeds.json --with_conf --feats input \
  --metric map50 --out $E/fig_main
```

### 5. (optional) Honest threshold pipeline
```bash
python method01_advantage_regress/get_thresholds.py \
  --dataset kitti --policy_glob "$O/sel_corr/policy_input_s*.pt" \
  --budgets 10,20,30,40,50,60,70,80,90 --name val_thresholds_mlp_backbone
```

### 6. Router overhead
```bash
python method01_advantage_regress/measure_router_overhead.py --group_dim 64 --hidden 128
# GAP-MLP: 60,241 params, 1.17e-4 GFLOPs (0.0004 % of SUPER)
```

## Results
See `outputs/kitti/eval/RESULTS.md`, `table_backbone_routercost.md`, and
`../ABLATION_PLAN.md` (full ablation roadmap).
