# Eval summary
run: 2026-06-03T16:50:21  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.62 | 48.5 | 3290.0 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.11 | 76.3 | 2092.0 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s4_t+010 | 45.4 | 21.15 | 0.6285 | +0.0008 | 60.5 | -19.6 | +24.8 | -19.9 |
| MAP | policy_input_s0_t+010 | 69.3 | 23.40 | 0.4149 | +0.0003 | 54.6 | -11.0 | +12.6 | -11.2 |