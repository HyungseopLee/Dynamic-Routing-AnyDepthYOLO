# Eval summary
run: 2026-06-02T18:32:25  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.82 | 48.0 | 3273.5 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.22 | 75.6 | 2079.3 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s3_t+010 | 53.7 | 21.94 | 0.6296 | +0.0020 | 57.8 | -16.6 | +20.4 | -16.9 |
| MAP | policy_input_s2_t+010 | 68.5 | 23.33 | 0.4152 | +0.0006 | 54.3 | -11.3 | +13.0 | -11.4 |