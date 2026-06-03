# Depth-Routing Policy — Ablation Study Plan

Frozen detector (AnyDepth-YOLOv12s). Only the policy (router) is trained, on the
offline cache (KITTI-detection train 5985 / val 1496). Evaluated on KITTI-tracking
video (21 seq, 8008 frames, 384x1248, conf=0.25). Detector BASE 16.87 / SUPER 26.30 GFLOPs.

**Protocol for every ablation (controlled variable):**
- vary ONE axis, hold the rest at the **default** (bold) below
- 5 seeds -> report mean +/- std band
- checkpoint selection = **val-corr** (fixed), loss = **MSE** (unless axis C)
- video metric = mAP50 / mAP@[.5:.95] vs SUPER usage (%) / GFLOPs
- router overhead measured separately (router_overhead.json), negligible (<0.001%)

---

## A. Input context features (what the router sees)
| id | axis | options | default | status |
|----|------|---------|---------|--------|
| A1 | feature source | backbone(input) / neck(pred) / both | **backbone** | ✅ done |
| A2 | tapped layers | backbone{4,6,8}, neck{14,17,20}, or subsets (e.g. only deepest) | **all 6** | ☐ |
| A3 | GAP normalization | none / **BN** / LN | (revisit) | ◐ partial (val-MSE era) |
| A4 | pooling | GAP / GAP+GMP / small spatial (k x k) | **GAP** | ☐ (A3 spatial costs FLOPs) |

> A3 must be re-run under val-corr (previous BN/LN/none used val-MSE and are stale).

## B. Policy network architecture
| id | axis | options | default | status |
|----|------|---------|---------|--------|
| B1 | prev-action embedding dim | none / (2,1) / (2,8) / (2,16) / (2,32) / (2,64) | **(2,8)** | ☐ |
| B2 | prev-action injection | concat / add / FiLM(scale-shift) | **concat** | ☐ |
| B3 | group projection dim | 16 / 32 / **64** / 128 | **64** | ☐ |
| B4 | head hidden dim | 64 / **128** / 256 | **128** | ☐ |
| B5 | head depth | 1 / **2** / 3 FC layers | **2** | ☐ |

## C. Training objective (loss on A-hat vs A)
| id | axis | options | default | status |
|----|------|---------|---------|--------|
| C1 | loss form | **MSE** / MAE / Huber / 1-Pearson(corr) / pairwise-ranking / Spearman-soft / BCE(sign A) | **MSE** | ☐ |
| C2 | advantage scaling | raw / **z-score(standardize)** | (regress uses raw) | ☐ |

> C1 "corr-as-loss" is the natural follow-up to the val-corr *selection* win.

## D. Checkpoint selection
| id | axis | options | default | status |
|----|------|---------|---------|--------|
| D1 | selection metric | val_mse / **val_corr** / last (plain) / last+reg | **val_corr** | ✅ done (backbone) |

> Result: val_mse << val_corr ~= plain-last; val_corr adopted (ranking-aligned, early stop ~ep50).

## E. Regularization
| id | axis | options | default | status |
|----|------|---------|---------|--------|
| E1 | dropout (head) | **0** / 0.1 / 0.2 / 0.3 | **0** | ◐ (reg+last tried) |
| E2 | weight decay | **0** / 1e-4 / 1e-3 | **0** | ◐ |
| E3 | epochs | 50 / 100 / **300** (val-corr picks ~50) | **300** | n/a |

## F. Temporal routing (eval-time)
| id | axis | options | default | status |
|----|------|---------|---------|--------|
| F1 | prev-action source | **recursive (t-1 chosen path)** / always-base feat | **recursive** | ✗ skipped |
| F2 | decision smoothing | none / hysteresis / EMA on A-hat | **none** | ☐ |

## (not ablation, just curve resolution)
- policy threshold count (policy_taus): 11 / 21 / 31 -- finer = smoother curve, same model.
- confidence baseline taus (conf_taus): granularity of the conf-routing baseline only.

---

## Suggested priority order
1. **A1 feature** ✅ (done — backbone wins)
2. **D1 selection** ✅ (done — val-corr)
3. **A3 normalization** (BN/LN/none) — re-run under val-corr  ← next
4. **B1 prev-action embed dim** (none..64) — cheap, directly user-requested
5. **C1 loss form** (MSE/MAE/Huber/corr/ranking) — most scientifically interesting
6. **B3/B4/B5 capacity** (group_dim/hidden/depth)
7. **B2 injection**, **A2 layers**, **F1/F2 temporal**, **E regularization**

Each is cache-based training (~3-4 min for 15 runs) + one ~20 min video eval.
