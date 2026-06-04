# method01 — GAP-MLP advantage-regression router (KITTI)

Baseline router variant: backbone feature → **GAP** → MLP → `Â` (predicted advantage).
This is the simplest router; method02 (tinyConv, spatial 2×2) is the main method and
Pareto-dominates this one (see `../../../method02_advantage_regress_tinyConv/outputs/kitti/eval/RESULTS.md`).

## Setup
- Detector: **AnyDepth-YOLOv12s, frozen**. 2-level depth: BASE 16.87 / SUPER 26.30 GFLOPs.
- Router: backbone layers {4,6,8} GAP → MLP → `Â`; loss = MSE(`Â`, `A=L_base−L_super`);
  **val-corr** checkpoint selection; **5 seeds** (mean±std).
- Train: offline cache (KITTI-detection 5985 train / 1496 val). Eval: KITTI-tracking
  **21 seq, 8008 frames**, 384×1248, conf=0.25 (all strategies same conf).
- Deploy: single `Â>τ` threshold sweeps the whole compute/accuracy curve (no retraining).

## Headline — backbone router vs always-SUPER (router cost included)

Source: `table_backbone_routercost.md`. #Params = detector 7.402M + router 0.060M.

| Depth Config | #Params | AP$_{50:95}$ | AP$_{50}$ | FLOPs | Energy | FPS | Latency |
|---|---|---|---|---|---|---|---|
| super (100%) | 7.402M | 41.46 | 62.77 | 26.30G | 3309 mJ | 48.2 | 20.76 ms |
| **Ours τ=0.1** (super 65.3%) | 7.462M | 41.25 | 62.74 | 23.03G | 2941 mJ | 54.3 | 18.44 ms |
| **Ours τ=0.2** (super 46.1%) | 7.462M | 41.06 | 62.55 | 21.21G | 2709 mJ | 58.9 | 16.98 ms |
| base (0%) | 7.402M | 39.31 | 59.13 | 16.87G | 2099 mJ | 76.0 | 13.17 ms |

- **τ=0.1**: AP50 ≈ always-SUPER (−0.03) → **−12 % FLOPs / −11 % energy / +13 % FPS**.
- **τ=0.2**: −0.22 AP50 → **−19 % FLOPs / −18 % energy / +22 % FPS**.
- Main figure (vs baselines): `fig_main_{ap50,ap5095}.png`. Baselines (random / luminance /
  edge / confidence routing) only reach SUPER accuracy near ~100 % SUPER (no savings).

## Ablations (all conf=0.25, 5 seeds, MSE + val-corr unless varied)

| Axis | Options | Conclusion | Figure / data |
|---|---|---|---|
| Feature source | backbone / neck / both | **backbone > both > neck** (backbone dominates low-FLOPs region) | `fig_feat_only_*`, `fig_ablation_feature_*`, `video_curve_featabl_seeds.json` |
| Prev-action embed dim | 0,1,8,16,32,64,128,768 | small consistent gain from including it; saturates by dim 8 (none < 8 ≈ 64) | `fig_ablation_pathdim_*`, `video_curve_B1pd{0,8,64,...}.json` |
| Prev-action mix prob (prev_p) | 0 / 0.25 / 0.5 / 0.75 / 1.0 | 0.5 (1:1) robust; extremes slightly worse | `fig_ablation_prevp_*`, `video_curve_Gprevp*.json` |
| Loss form | MSE / MAE / Huber / corr | MSE best/at-par; ranking-corr no gain | `fig_ablation_loss_*`, `video_curve_C1{corr,huber,mae}.json` |
| Checkpoint selection | val-corr / plain-last / reg-last | **val-corr** (ranking-aligned); plain-last ≈ val-corr | `fig_ablation_select_*`, `video_curve_sel*.json` |
| GAP normalization | none / BN / LN | **no significant difference** (bands overlap) | `fig_ablation_norm_*`, `video_curve_A3{none,layer}.json` |

## Honest threshold pipeline (optional)
`val_thresholds_mlp_backbone.json` (get_thresholds.py): τ fixed on val per FLOPs budget,
applied only on video → `video_curve_mlp_backbone_valtau.json`.

## Files
- Headline table: `table_backbone_routercost.md` (data: `video_curve_backbone_routercost.json`)
- Main figure: `fig_main_*` ; feature: `fig_feat_only_*` ; ablations: `fig_ablation_*`
- Router cost: `../router_overhead.json` (router ≈1.2e-4 GFLOPs, <0.001 % of SUPER)
- Roadmap: `../../ABLATION_PLAN.md`
