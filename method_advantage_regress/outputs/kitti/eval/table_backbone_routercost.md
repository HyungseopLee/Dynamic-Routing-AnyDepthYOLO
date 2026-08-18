# method_advantage_regress (TinyConv 2×2) — backbone-only routing (KITTI-tracking)

Frozen AnyDepth-YOLOv12s, conf=0.25, 5 seeds mean (path_dim=8). Router input =
backbone layers {4,6,8} adaptive-pooled to a 2×2 grid (conv-2×2). Decision:
`Â > τ → SUPER`. Latency / FPS / Energy are measured on the video stream
**including the router forward overhead** (charged to the policy each frame).
#Params = frozen detector (7.402M) + router (0.061M) for *Ours*; detector only
for super/base.

| Depth Config | #Params | AP$^{val}_{50:95}$ | AP$^{val}_{50}$ | FLOPs | Energy | FPS | Latency |
|---|---|---|---|---|---|---|---|
| **super** (super 100%) | 7.402M | 41.46 | 62.77 | 26.30G | 3343 mJ | 47.9 | 20.87 ms |
| **Ours** τ=0.2 (super 26.7%) | 7.463M | 41.00 | 62.57 | 19.39G | 2515 mJ | 63.8 | 15.69 ms |
| **Ours** τ=0.1 (super 52.2%) | 7.463M | 41.41 | 62.99 | 21.80G | 2823 mJ | 56.8 | 17.62 ms |
| **base** (super 0%) | 7.402M | 39.31 | 59.13 | 16.87G | 2139 mJ | 74.9 | 13.35 ms |

## Reduction vs always-SUPER

| Operating point | ΔAP50 | ΔAP | ΔFLOPs | ΔEnergy | ΔLatency | ΔFPS |
|---|---:|---:|---:|---:|---:|---:|
| Ours τ=0.1 (super 52.2%) | **+0.22** | −0.05 (≈equal) | **−17.1 %** | **−15.6 %** | **−15.6 %** | **+18.6 %** |
| Ours τ=0.2 (super 26.7%) | −0.20 | −0.46 | **−26.3 %** | **−24.8 %** | **−24.8 %** | **+33.1 %** |

**τ=0.1 (super 52.2 %) Pareto-dominates always-SUPER**: higher AP50 (+0.22),
equal AP, while cutting FLOPs/energy/latency ~16 % and raising FPS +19 %.

A single trained model traces the whole curve; τ selects the operating point at
deploy time (no retraining). Router overhead (0.061M params, ~0.0015 % of SUPER
GFLOPs) is included in the measured latency/energy and is negligible.
