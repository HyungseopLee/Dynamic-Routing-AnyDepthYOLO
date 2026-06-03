# Eval summary
run: 2026-06-02T19:25:47  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.96 | 47.7 | 3296.4 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.31 | 75.1 | 2093.8 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s2_t+020 | 58.7 | 22.41 | 0.6286 | +0.0009 | 56.2 | -14.8 | +17.8 | -15.0 |
| MAP | policy_input_s3_t+010 | 65.2 | 23.02 | 0.4155 | +0.0009 | 54.7 | -12.5 | +14.6 | -12.7 |