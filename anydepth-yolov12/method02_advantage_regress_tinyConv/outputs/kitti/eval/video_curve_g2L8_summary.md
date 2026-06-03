# Eval summary
run: 2026-06-03T18:16:21  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.53 | 48.7 | 3340.3 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.04 | 76.7 | 2120.8 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_t+010 | 40.7 | 20.71 | 0.6283 | +0.0007 | 62.2 | -21.3 | +27.7 | -21.7 |
| MAP | policy_input_s1_t-010 | 96.1 | 25.93 | 0.4146 | +0.0000 | 49.4 | -1.4 | +1.5 | -1.5 |