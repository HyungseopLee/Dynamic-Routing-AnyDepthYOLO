# Eval summary
run: 2026-06-03T23:48:30  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.97 | 47.7 | 3407.1 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.43 | 74.5 | 2182.6 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s3_t+010 | 34.7 | 20.14 | 0.6282 | +0.0005 | 62.3 | -23.4 | +30.7 | -23.4 |
| MAP | policy_input_s4_t+000 | 65.7 | 23.07 | 0.4158 | +0.0012 | 54.4 | -12.3 | +14.0 | -12.3 |