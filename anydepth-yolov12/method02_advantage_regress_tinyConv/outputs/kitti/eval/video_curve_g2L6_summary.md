# Eval summary
run: 2026-06-03T18:08:35  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.84 | 48.0 | 3363.4 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.27 | 75.4 | 2142.0 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s2_t+020 | 36.6 | 20.32 | 0.6286 | +0.0009 | 62.5 | -22.7 | +30.2 | -23.1 |
| MAP | policy_input_s1_t+010 | 67.3 | 23.22 | 0.4155 | +0.0009 | 54.5 | -11.7 | +13.6 | -12.0 |