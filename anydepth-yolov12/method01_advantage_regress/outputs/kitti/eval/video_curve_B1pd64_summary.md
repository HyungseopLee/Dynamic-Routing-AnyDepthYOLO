# Eval summary
run: 2026-06-02T16:07:15  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.79 | 48.1 | 3298.8 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.19 | 75.8 | 2093.5 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s2_t+010 | 61.6 | 22.68 | 0.6283 | +0.0006 | 55.9 | -13.8 | +16.3 | -13.9 |
| MAP | policy_input_s4_t+010 | 63.7 | 22.88 | 0.4149 | +0.0003 | 55.5 | -13.0 | +15.3 | -13.2 |