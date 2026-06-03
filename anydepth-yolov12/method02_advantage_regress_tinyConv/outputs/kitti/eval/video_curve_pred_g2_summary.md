# Eval summary
run: 2026-06-03T21:39:55  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 21.67 | 46.1 | 3659.5 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.97 | 71.6 | 2363.4 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_t+010 | 77.5 | 24.18 | 0.6292 | +0.0015 | 50.1 | -8.1 | +8.6 | -7.7 |
| MAP | policy_input_s0_t+010 | 77.5 | 24.18 | 0.4151 | +0.0005 | 50.1 | -8.1 | +8.6 | -7.7 |