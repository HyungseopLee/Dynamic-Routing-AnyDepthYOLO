# Eval summary
run: 2026-06-22T18:01:57  |  sequences=21 frames=8008 strategies=137 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6276 | 0.4146 | 26.30 | 19.68 | 50.8 | 3076.4 |
| always-BASE | 0.5913 | 0.3930 | 16.87 | 12.57 | 79.5 | 1965.4 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_both_p05_s3_b40 | 40.9 | 20.73 | 0.6283 | +0.0006 | 62.9 | -21.2 | +23.9 | -19.3 |
| MAP | policy_both_p05_s3_b70 | 66.7 | 23.16 | 0.4148 | +0.0002 | 56.4 | -11.9 | +11.0 | -10.0 |