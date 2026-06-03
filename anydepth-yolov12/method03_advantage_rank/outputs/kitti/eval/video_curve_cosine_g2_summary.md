# Eval summary
run: 2026-06-03T23:32:31  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.81 | 48.1 | 3354.9 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.33 | 75.0 | 2149.4 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s1_t+020 | 40.6 | 20.70 | 0.6302 | +0.0025 | 61.0 | -21.3 | +27.0 | -21.2 |
| MAP | policy_input_s1_t+010 | 69.6 | 23.43 | 0.4159 | +0.0013 | 53.9 | -10.9 | +12.3 | -10.9 |