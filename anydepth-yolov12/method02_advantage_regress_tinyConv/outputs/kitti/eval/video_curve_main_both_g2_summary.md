# Eval summary
run: 2026-06-15T15:38:16  |  sequences=21 frames=8008 strategies=194 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.69 | 48.3 | 3276.5 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.23 | 75.6 | 2096.2 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_both_s1_t+020 | 28.2 | 19.53 | 0.6285 | +0.0009 | 63.4 | -25.7 | +31.2 | -23.7 |
| MAP | policy_both_s4_t+010 | 67.7 | 23.25 | 0.4150 | +0.0004 | 53.5 | -11.6 | +10.6 | -9.5 |