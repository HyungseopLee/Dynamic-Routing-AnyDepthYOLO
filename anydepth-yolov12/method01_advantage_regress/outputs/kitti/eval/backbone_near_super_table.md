# Backbone-only policy: operating points matching / exceeding always-SUPER (KITTI-tracking)

Setup: AnyDepth-YOLOv12s (frozen), KITTI-tracking 21 seq / 8008 frames, 384x1248,
conf=0.25. Policy: backbone(input) feature, val-corr selection, regress (MSE),
mean over 5 seeds. Router overhead negligible (<0.001% GFLOPs, see router_overhead.json).

Reference endpoints:
| config | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.61 | 48.5 | 3115.8 |
| always-BASE  | 0.5913 | —      | 16.87 | 13.10 | 76.3 | 1980.6 |

Backbone-policy points with mAP50 >= (super - 0.002), i.e. statistically on par or above:

| policy τ (Â) | SUPER % | GFLOPs | mAP50 | ΔmAP50 vs super | mAP | latency(ms) | FPS | energy(mJ) | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| +0.20 | 48.2 | 21.42 | 0.6273 | −0.0004 | 0.4129 | 16.72 | 59.9 | 2531.3 | **−18.6%** | **+23.5%** | **−18.8%** |
| +0.10 | 70.0 | 23.48 | **0.6303** | **+0.0027** | 0.4150 | 18.36 | 54.5 | 2777.3 | −10.7% | +12.4% | −10.9% |
| +0.00 | 87.6 | 25.13 | 0.6288 | +0.0012 | 0.4150 | 19.68 | 50.8 | 2975.3 | −4.4% | +4.8% | −4.5% |
| −0.10 | 95.9 | 25.92 | 0.6280 | +0.0003 | 0.4148 | 20.30 | 49.3 | 3069.7 | −1.5% | +1.5% | −1.5% |
| −0.20 | 98.6 | 26.17 | 0.6278 | +0.0001 | 0.4147 | 20.50 | 48.8 | 3099.5 | −0.5% | +0.5% | −0.5% |

Highlights:
- **τ=+0.10**: *exceeds* always-SUPER mAP50 (+0.0027) and mAP (+0.0004) while using only
  70% SUPER -> **−10.7% FLOPs, +12.4% FPS, −10.9% energy**.
- **τ=+0.20**: matches super mAP50 (−0.0004, within noise) at 48% SUPER ->
  **−18.6% FLOPs, +23.5% FPS, −18.8% energy** — biggest efficiency win at ~equal AP.
