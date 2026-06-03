# Eval summary
run: 2026-06-03T21:47:50  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.71 | 48.3 | 3318.9 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.16 | 76.0 | 2108.9 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s1_t+020 | 28.2 | 19.53 | 0.6285 | +0.0009 | 65.4 | -25.7 | +35.4 | -26.1 |
| MAP | policy_input_s4_t+010 | 67.7 | 23.25 | 0.4150 | +0.0004 | 54.8 | -11.6 | +13.5 | -11.9 |