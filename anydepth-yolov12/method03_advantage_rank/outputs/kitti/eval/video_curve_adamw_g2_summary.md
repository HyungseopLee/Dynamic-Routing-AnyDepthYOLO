# Eval summary
run: 2026-06-03T23:40:33  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 21.32 | 46.9 | 3454.6 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.63 | 73.4 | 2209.5 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_t+010 | 37.4 | 20.40 | 0.6281 | +0.0005 | 60.6 | -22.5 | +29.3 | -22.6 |
| MAP | policy_input_s2_t+010 | 58.1 | 22.35 | 0.4152 | +0.0006 | 55.4 | -15.0 | +18.0 | -15.2 |