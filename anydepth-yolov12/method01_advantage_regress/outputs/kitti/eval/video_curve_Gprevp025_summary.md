# Eval summary
run: 2026-06-02T19:15:58  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 21.21 | 47.2 | 3761.0 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.62 | 73.4 | 2420.2 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s2_t+020 | 37.8 | 20.43 | 0.6283 | +0.0006 | 60.6 | -22.3 | +28.6 | -21.9 |
| MAP | policy_input_s3_t+000 | 88.3 | 25.20 | 0.4149 | +0.0003 | 49.2 | -4.2 | +4.4 | -3.9 |