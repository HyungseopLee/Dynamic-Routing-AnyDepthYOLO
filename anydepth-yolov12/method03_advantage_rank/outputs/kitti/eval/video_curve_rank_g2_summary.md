# Eval summary
run: 2026-06-03T22:50:36  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 22.54 | 44.4 | 4397.9 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 15.07 | 66.4 | 3013.8 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_t+020 | 53.4 | 21.91 | 0.6287 | +0.0010 | 52.4 | -16.7 | +18.2 | -14.3 |
| MAP | policy_input_s4_t+000 | 59.4 | 22.47 | 0.4149 | +0.0003 | 51.3 | -14.6 | +15.6 | -12.2 |