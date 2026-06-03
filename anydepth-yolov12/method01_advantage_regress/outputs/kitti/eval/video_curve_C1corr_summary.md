# Eval summary
run: 2026-06-02T18:51:51  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 21.05 | 47.5 | 3309.7 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.40 | 74.6 | 2108.0 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s1_t+010 | 31.4 | 19.83 | 0.6277 | +0.0000 | 63.3 | -24.6 | +33.1 | -24.9 |
| MAP | policy_input_s1_t-010 | 71.3 | 23.60 | 0.4151 | +0.0005 | 52.9 | -10.3 | +11.3 | -10.2 |