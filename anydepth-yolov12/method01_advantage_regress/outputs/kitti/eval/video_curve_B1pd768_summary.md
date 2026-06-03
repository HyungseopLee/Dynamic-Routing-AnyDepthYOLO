# Eval summary
run: 2026-06-02T20:35:47  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 21.06 | 47.5 | 3305.4 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.40 | 74.6 | 2103.3 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s0_t+020 | 50.0 | 21.59 | 0.6292 | +0.0015 | 58.0 | -17.9 | +22.3 | -18.2 |
| MAP | policy_input_s4_t+020 | 61.2 | 22.64 | 0.4151 | +0.0005 | 55.3 | -13.9 | +16.5 | -14.1 |