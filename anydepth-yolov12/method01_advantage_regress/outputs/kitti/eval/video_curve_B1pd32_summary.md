# Eval summary
run: 2026-06-02T15:57:28  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.59 | 48.6 | 3272.7 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.07 | 76.5 | 2077.8 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s4_t+030 | 28.5 | 19.56 | 0.6285 | +0.0008 | 65.7 | -25.6 | +35.4 | -26.0 |
| MAP | policy_input_s3_t+000 | 87.1 | 25.08 | 0.4152 | +0.0006 | 51.0 | -4.6 | +5.0 | -4.7 |