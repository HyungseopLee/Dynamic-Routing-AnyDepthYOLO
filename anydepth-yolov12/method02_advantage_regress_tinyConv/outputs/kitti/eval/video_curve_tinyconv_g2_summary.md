# Eval summary
run: 2026-06-03T17:27:16  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.57 | 48.6 | 3310.8 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.09 | 76.4 | 2106.5 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_t+010 | 41.0 | 20.74 | 0.6280 | +0.0003 | 61.9 | -21.2 | +27.2 | -21.4 |
| MAP | policy_input_s2_t+010 | 54.5 | 22.01 | 0.4167 | +0.0021 | 58.2 | -16.3 | +19.8 | -16.5 |