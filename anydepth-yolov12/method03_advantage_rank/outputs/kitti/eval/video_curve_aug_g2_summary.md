# Eval summary
run: 2026-06-03T23:56:36  |  sequences=21 frames=8008 strategies=107 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 21.49 | 46.5 | 3483.2 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.79 | 72.5 | 2235.5 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s2_t+010 | 40.4 | 20.69 | 0.6278 | +0.0001 | 59.1 | -21.4 | +27.1 | -21.3 |
| MAP | policy_input_s3_t+000 | 75.1 | 23.95 | 0.4148 | +0.0002 | 51.1 | -8.9 | +9.8 | -9.0 |