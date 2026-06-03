# Eval summary
run: 2026-06-02T19:36:31  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.68 | 48.4 | 3254.3 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.12 | 76.2 | 2064.8 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s3_t+020 | 46.1 | 21.22 | 0.6278 | +0.0002 | 60.2 | -19.3 | +24.5 | -19.7 |
| MAP | policy_input_s3_t+010 | 66.8 | 23.17 | 0.4152 | +0.0006 | 55.0 | -11.9 | +13.8 | -12.1 |