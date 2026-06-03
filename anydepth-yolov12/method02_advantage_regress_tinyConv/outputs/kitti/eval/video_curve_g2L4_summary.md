# Eval summary
run: 2026-06-03T18:00:41  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.66 | 48.4 | 3312.9 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.14 | 76.1 | 2106.4 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s3_t+030 | 49.0 | 21.49 | 0.6294 | +0.0017 | 59.5 | -18.3 | +22.9 | -18.6 |
| MAP | policy_input_s2_t+010 | 69.9 | 23.46 | 0.4165 | +0.0019 | 54.4 | -10.8 | +12.4 | -11.0 |