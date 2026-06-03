# Eval summary
run: 2026-06-02T15:38:09  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.77 | 48.1 | 3300.0 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.16 | 76.0 | 2090.9 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s4_t+020 | 51.3 | 21.71 | 0.6284 | +0.0007 | 58.7 | -17.5 | +21.9 | -17.9 |
| MAP | policy_input_s4_t+000 | 84.3 | 24.82 | 0.4155 | +0.0009 | 51.2 | -5.6 | +6.3 | -5.9 |