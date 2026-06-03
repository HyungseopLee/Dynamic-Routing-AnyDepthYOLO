# Eval summary
run: 2026-06-03T15:34:52  |  sequences=21 frames=8008 strategies=47 conf=0.001
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.7032 | 0.4549 | 26.30 | 20.69 | 48.3 | 3057.2 |
| always-BASE | 0.6725 | 0.4370 | 16.87 | 13.68 | 73.1 | 2022.0 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_b40 | 55.8 | 22.14 | 0.7036 | +0.0005 | 56.9 | -15.8 | +17.7 | -15.0 |
| MAP | policy_input_s0_b20 | 39.9 | 20.63 | 0.4549 | +0.0000 | 60.7 | -21.6 | +25.6 | -20.3 |