# Eval summary
run: 2026-06-02T19:46:11  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.65 | 48.4 | 3253.2 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.10 | 76.3 | 2064.5 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s2_t+010 | 78.5 | 24.28 | 0.6291 | +0.0014 | 52.6 | -7.7 | +8.5 | -7.9 |
| MAP | policy_input_s1_t-010 | 96.7 | 26.00 | 0.4149 | +0.0002 | 49.0 | -1.2 | +1.2 | -1.2 |