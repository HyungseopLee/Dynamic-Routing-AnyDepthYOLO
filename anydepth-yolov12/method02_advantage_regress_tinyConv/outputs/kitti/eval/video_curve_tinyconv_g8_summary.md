# Eval summary
run: 2026-06-03T17:35:00  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.57 | 48.6 | 3327.6 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.04 | 76.7 | 2109.6 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s4_t+010 | 63.2 | 22.83 | 0.6282 | +0.0005 | 56.2 | -13.2 | +15.5 | -13.4 |
| MAP | policy_input_s0_t+010 | 65.5 | 23.05 | 0.4147 | +0.0001 | 55.6 | -12.4 | +14.4 | -12.6 |