# method01 (GAP-MLP) — backbone-only routing (KITTI-tracking)

Frozen AnyDepth-YOLOv12s, conf=0.25, 5 seeds mean (path_dim=8). Router input =
backbone layers {4,6,8} GAP. Decision: `Â > τ → SUPER`. Latency / FPS / Energy
are measured on the video stream **including the router forward overhead**
(charged to the policy each frame). #Params = frozen detector (7.402M) + router
(0.060M) for *Ours*; detector only for super/base.

| Depth Config | #Params | AP$^{val}_{50:95}$ | AP$^{val}_{50}$ | FLOPs | Energy | FPS | Latency |
|---|---|---|---|---|---|---|---|
| **super** (super 100%) | 7.402M | 41.46 | 62.77 | 26.30G | 3309 mJ | 48.2 | 20.76 ms |
| **Ours** τ=0.2 (super 46.1%) | 7.462M | 41.06 | 62.55 | 21.21G | 2709 mJ | 58.9 | 16.98 ms |
| **Ours** τ=0.1 (super 65.3%) | 7.462M | 41.25 | 62.74 | 23.03G | 2941 mJ | 54.3 | 18.44 ms |
| **base** (super 0%) | 7.402M | 39.31 | 59.13 | 16.87G | 2099 mJ | 76.0 | 13.17 ms |

## Reduction vs always-SUPER

| Operating point | ΔAP50 | ΔAP | ΔFLOPs | ΔEnergy | ΔLatency | ΔFPS |
|---|---:|---:|---:|---:|---:|---:|
| Ours τ=0.1 (super 65.3%) | −0.03 (≈equal) | −0.21 | **−12.4 %** | **−11.1 %** | **−11.2 %** | **+12.7 %** |
| Ours τ=0.2 (super 46.1%) | −0.22 | −0.40 | **−19.4 %** | **−18.1 %** | **−18.2 %** | **+22.2 %** |

A single trained model traces the whole curve; τ selects the operating point at
deploy time (no retraining). Router overhead (0.060M params, <0.0005 % of SUPER
GFLOPs) is included in the measured latency/energy and is negligible.
