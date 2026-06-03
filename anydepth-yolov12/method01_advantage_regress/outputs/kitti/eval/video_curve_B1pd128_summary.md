# Eval summary
run: 2026-06-02T20:25:53  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.96 | 47.7 | 3288.1 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.28 | 75.3 | 2083.3 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_t+020 | 36.9 | 20.35 | 0.6282 | +0.0006 | 62.2 | -22.6 | +30.3 | -23.2 |
| MAP | policy_input_s4_t+000 | 91.1 | 25.46 | 0.4147 | +0.0001 | 49.3 | -3.2 | +3.4 | -3.2 |