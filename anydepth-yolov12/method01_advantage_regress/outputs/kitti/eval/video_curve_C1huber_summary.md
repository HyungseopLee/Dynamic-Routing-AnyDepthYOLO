# Eval summary
run: 2026-06-02T18:42:07  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.53 | 48.7 | 3235.9 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.01 | 76.9 | 2050.9 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_t+020 | 53.0 | 21.87 | 0.6304 | +0.0027 | 58.8 | -16.9 | +20.7 | -17.1 |
| MAP | policy_input_s2_t+000 | 86.8 | 25.06 | 0.4151 | +0.0005 | 51.2 | -4.7 | +5.1 | -4.9 |