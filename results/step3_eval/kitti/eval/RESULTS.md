# method_advantage_regress — TinyConv advantage-regression router (KITTI) — MAIN METHOD

Router: backbone layers {4,6,8} adaptive-pooled to a **2×2 grid** → tiny spatial conv
(1×1 C→d, ReLU, 3×3 depthwise, ReLU, GAP) → MLP → `Â`. Keeps coarse spatial layout
("where in the frame the hard content is") instead of collapsing it like GAP-MLP (method01).
Decision: `Â>τ → SUPER`; one trained model + scalar τ traces the full Pareto curve.

## Setup
- Detector: **AnyDepth-YOLOv12s, frozen**. BASE 16.87 / SUPER 26.30 GFLOPs.
- Loss MSE(`Â`,`A=L_base−L_super`); **val-corr** selection; **5 seeds**; conf=0.25.
- Eval: KITTI-tracking 21 seq / 8008 frames, 384×1248. Latency/FPS/Energy measured on the
  video stream **including the router forward overhead**.

## Headline — conv-2×2 backbone router vs always-SUPER (router cost included)

Source: `table_backbone_routercost.md`. #Params = detector 7.402M + router 0.061M.

| Depth Config | #Params | AP$_{50:95}$ | AP$_{50}$ | FLOPs | Energy | FPS | Latency |
|---|---|---|---|---|---|---|---|
| super (100%) | 7.402M | 41.46 | 62.77 | 26.30G | 3343 mJ | 47.9 | 20.87 ms |
| **Ours τ=0.1** (super 52.2%) | 7.463M | **41.41** | **62.99** | 21.80G | 2823 mJ | 56.8 | 17.62 ms |
| **Ours τ=0.2** (super 26.7%) | 7.463M | 41.00 | 62.57 | 19.39G | 2515 mJ | 63.8 | 15.69 ms |
| base (0%) | 7.402M | 39.31 | 59.13 | 16.87G | 2139 mJ | 74.9 | 13.35 ms |

- **τ=0.1 Pareto-dominates always-SUPER**: AP50 **+0.22** (62.99 vs 62.77), AP equal,
  while **−17 % FLOPs / −16 % energy / −16 % latency / +19 % FPS**.
- **τ=0.2**: −0.20 AP50 at only 26.7 % SUPER → **−26 % FLOPs / −25 % energy / +33 % FPS**.
- Full threshold sweep with reductions: `tinyconv_g2_backbone_table.md`.
- Main figure (vs baselines): `fig_main_g2_{ap50,ap5095}.png`.

## vs method01 (GAP-MLP) — why spatial conv
At τ=0.1, TinyConv reaches the same AP50 as GAP-MLP using **52 % SUPER vs 65 %**, i.e. it
identifies the hard frames more precisely and spends less compute (−16 % vs −11 % energy).
Comparison figure: `fig_conv_vs_gapmlp_g2_*`.

## Ablations (conv-2×2 backbone default; conf=0.25, 5 seeds)

| Axis | Options | Conclusion | Figure / data |
|---|---|---|---|
| Router design | GAP-MLP vs **conv-2×2** | **conv > GAP-MLP** (spatial layout helps) | `fig_conv_vs_gapmlp_g2_*` |
| Grid size | 2×2 / 4×4 / 8×8 | **2×2 ≈ 4×4 ≈ 8×8** → resolution beyond 2×2 doesn't help (2×2 = sweet spot) | `fig_grid_ablation_*`, `video_curve_tinyconv_{g2,backbone(=g4),g8}.json` |
| Grid shape (equal cells) | square vs aspect-rect | **square > rect**; collapsing vertical (1×4) hurts most | `fig_gridshape_{4cell,16cell}_*`, `video_curve_g{1x4,2x8}.json` |
| Backbone level | L4 / L6 / L8 (each alone) | **deep L6/L8 ≫ shallow L4** | `fig_level_ablation_*`, `video_curve_g2L{4,6,8}.json` |
| Feature source | backbone / neck / both | **backbone > both > neck** | `fig_feat_ablation_g2_*`, `video_curve_{tinyconv_g2,pred_g2,both_g2}.json` |
| Prev-action embed | with (dim 8) / without | small consistent gain at the operating budget (~+0.0015 AP50 @ 50 % SUPER) | `fig_prevaction_ablation_g2_*`, `video_curve_noprev_g2.json` |

## Router overhead (per frame, batch=1)
| Router | #Params | GFLOPs | % of SUPER |
|---|---|---|---|
| GAP-MLP (method01) | 60,241 | 1.17e-4 | 0.0004 % |
| **TinyConv 2×2** | 60,881 | 4.17e-4 | 0.0015 % |
| TinyConv 4×4 | 60,881 | 1.61e-3 | 0.0060 % |

All negligible → router choice is driven purely by accuracy. Source: `../router_overhead.json`.

## Honest threshold pipeline (optional)
`val_thresholds_tinyconv_backbone.json` → `video_curve_tinyconv_backbone_valtau.json`
(τ fixed on val per budget, applied only on video).

## Key result jsons
- conv-2×2 backbone (main): `video_curve_tinyconv_g2.json`, router-cost `video_curve_g2_routercost.json`
- main merged (with baselines): `video_curve_main_g2_merged.json`
