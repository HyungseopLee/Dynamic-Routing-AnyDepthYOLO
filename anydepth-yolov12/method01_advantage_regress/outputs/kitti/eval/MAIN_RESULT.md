# Main Result — AnyDepth Depth Routing (KITTI-tracking)

## Setup
- Detector: AnyDepth-YOLOv12s, **frozen**. 2-level depth: BASE (essential) / SUPER (full).
- Policy (router): backbone(input) GAP feature -> MLP -> A-hat (predicted advantage),
  threshold at eval. regress loss MSE(A-hat, A); **val-corr** checkpoint selection; 5 seeds.
- Train: offline cache (KITTI-detection, 5985 train / 1496 val). Eval: KITTI-tracking
  **21 sequences, 8008 frames**, 384x1248, conf=0.25 (all strategies same conf).
- Detector cost: **BASE 16.87 / SUPER 26.30 GFLOPs**. Router overhead negligible
  (~1.2e-4 GFLOPs, <0.001%, see router_overhead.json).

## Headline (backbone policy vs always-SUPER)

| Method | AP50 | AP50:95 | GFLOPs | Latency(ms) | FPS | Energy(mJ) |
|---|---|---|---|---|---|---|
| Always-SUPER | 0.628 | 0.415 | 26.3 | 20.6 | 48.5 | 3116 |
| **Ours (τ=+0.2)** | **0.627** | 0.413 | **21.4 (−19%)** | **16.7 (−19%)** | **59.9 (+24%)** | **2531 (−19%)** |
| **Ours (τ=+0.1)** | 0.630 | **0.415** | **23.5 (−11%)** | **18.4 (−11%)** | **54.5 (+12%)** | **2777 (−11%)** |
| Always-BASE | 0.591 | 0.393 | 16.9 | 13.1 | 76.3 | 1981 |

- **τ=+0.2**: matches AP50 (0.627≈0.628) at 48% SUPER → −19% FLOPs / +24% FPS / −19% energy.
- **τ=+0.1**: matches/exceeds both AP50 (0.630) and AP50:95 (0.415) at 70% SUPER → −11% FLOPs / +12% FPS / −11% energy.

Takeaway: at **equal accuracy**, the router cuts FLOPs/energy ~11-19% and raises FPS
12-24%. Baselines (random / luminance / edge / confidence routing) reach super accuracy
only near ~100% SUPER (no savings). See main_result_{map50,map}.png.

## Findings
- Feature ablation: **backbone > both > neck** (backbone dominates the low-FLOPs region).
- Selection: val-MSE early-stop picks an underfit epoch (sparse curve); **val-corr** fixes
  it (ranking-aligned) — adopted. plain-last ~= val-corr (overfitting on image-MSE does
  not hurt video AP).
- Normalization (none/BN1d/LN): no significant difference (bands overlap).

## Files
- Data: video_curve_featabl_seeds.json (+ .log evidence)
- Figures: ablation_seeds_{map50,map}.png, main_result_{map50,map}.png
- Tables: backbone_near_super_table.md, video_curve_featabl_seeds_super_table.md
- Router cost: ../router_overhead.json | Ablation roadmap: ../../ABLATION_PLAN.md

## Ablation status (see ABLATION_PLAN.md)
- A1 feature ✅ | D1 selection (val-corr) ✅ | A3 normalization ✅ (no effect)
- B1 prev-action embed dim ◐ (running) | C1 loss form ☐ | capacity/temporal ☐
