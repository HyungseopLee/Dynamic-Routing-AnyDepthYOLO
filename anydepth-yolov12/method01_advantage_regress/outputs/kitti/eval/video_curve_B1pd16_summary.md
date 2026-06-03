# Eval summary
run: 2026-06-02T15:47:47  |  sequences=21 frames=8008 strategies=156 conf=0.25
detector: BASE 16.87 / SUPER 26.30 GFLOPs

| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |
|---|---|---|---|---|---|---|
| always-SUPER | 0.6277 | 0.4146 | 26.30 | 20.61 | 48.5 | 3280.6 |
| always-BASE | 0.5913 | 0.3931 | 16.87 | 13.06 | 76.6 | 2079.0 |

Best policy point matching/exceeding always-SUPER (lowest GFLOPs):
| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |
|---|---|---|---|---|---|---|---|---|---|
| MAP50 | policy_input_s4_t+020 | 42.3 | 20.86 | 0.6293 | +0.0016 | 61.6 | -20.7 | +26.9 | -21.0 |
| MAP | policy_input_s2_t+010 | 62.4 | 22.76 | 0.4155 | +0.0009 | 56.3 | -13.5 | +16.1 | -13.8 |