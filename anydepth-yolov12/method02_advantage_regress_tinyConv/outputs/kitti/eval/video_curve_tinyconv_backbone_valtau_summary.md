# Eval summary
run: 2026-06-03T15:14:57  |  sequences=21 frames=8008 strategies=47 conf=0.001
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.7032 | 0.4549 | 26.30 | 20.76 | 48.2 | 3502.7 |
| always-BASE | 0.6725 | 0.4370 | 16.87 | 13.66 | 73.2 | 2305.4 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s3_b30 | 41.9 | 20.82 | 0.7033 | +0.0001 | 60.4 | -20.8 | +25.4 | -20.0 |
| MAP | policy_input_s4_b30 | 43.1 | 20.94 | 0.4550 | +0.0001 | 60.0 | -20.4 | +24.6 | -19.9 |