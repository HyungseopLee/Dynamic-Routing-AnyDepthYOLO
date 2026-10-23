"""Does the learned router preferentially route the frames where SUPER matters most?

Comment 3 (reviewer): global mAP/AR averages may conceal single-frame missed
detections. This computes, per frame and across ALL classes (not just safety-
critical -- that split is handled separately by the risk-weighted loss ablation),
the BASE->SUPER precision and recall gap at a fixed confidence operating point
(conf=0.25, matching the deployment threshold used elsewhere in this repo; frames.npz
itself is dumped at conf=0.001 to support the full AP curve, so we re-threshold here).
IoU=0.5 is used for the TP flags (index 0 of the 10-point 0.50:0.05:0.95 grid).

Frames are then sorted by recall_gap (primary) and precision_gap (secondary)
descending -- i.e. the frames where switching BASE->SUPER helps the most. At each
target SUPER budget, we ask: of the top-K% frames by gap (K = budget), what fraction
does the ACTUAL router (tau solved to hit that budget, 5 seeds averaged) send to
SUPER, versus what a random router at the same budget would be expected to catch by
chance (= the budget itself)?

Usage (run from repo root):
    python -m step3_eval.ablation.frame_gap_coverage \
        --dump results/step3_eval/kitti/eval/frames.npz \
        --out results/step3_eval/kitti/eval/frame_gap_coverage.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from step3_eval.ablation.tau_sweep import load, decide, solve_tau

CONF = 0.25
IOU_IDX = 0  # IoU=0.50


def per_frame_precision_recall(d, path, conf_thres=CONF):
    """Per-frame (precision, recall) at a fixed confidence threshold, all classes."""
    off = d[f"off_{path}"]
    conf = d[f"conf_{path}"]
    tp = d[f"tp_{path}"][:, IOU_IDX]
    gt_per_frame = d["gt_count"].sum(1).astype(np.float64)

    n = len(off) - 1
    prec = np.full(n, np.nan)
    rec = np.full(n, np.nan)
    for i in range(n):
        s, e = off[i], off[i + 1]
        keep = conf[s:e] >= conf_thres
        ndet = int(keep.sum())
        ntp = int(tp[s:e][keep].sum())
        gt = gt_per_frame[i]
        if ndet > 0:
            prec[i] = ntp / ndet
        if gt > 0:
            rec[i] = ntp / gt
    return prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--targets", default="10,30,50,70,90")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load(args.dump)
    n = len(d["seq"])
    tags = d["tags"]

    p_base, r_base = per_frame_precision_recall(d, "base", args.conf)
    p_super, r_super = per_frame_precision_recall(d, "super", args.conf)

    prec_gap = p_super - p_base
    rec_gap = r_super - r_base

    has_gt = d["gt_count"].sum(1) > 0
    print(f"[*] {n} frames, {int(has_gt.sum())} with GT>0 "
          f"(recall_gap defined on these)")
    print(f"[*] recall_gap: mean={np.nanmean(rec_gap):+.4f} "
          f"median={np.nanmedian(rec_gap):+.4f} "
          f">0: {(rec_gap > 0).sum()}  <0: {(rec_gap < 0).sum()}  "
          f"==0/nan: {n - (rec_gap > 0).sum() - (rec_gap < 0).sum()}")
    print(f"[*] precision_gap: mean={np.nanmean(prec_gap):+.4f} "
          f"median={np.nanmedian(prec_gap):+.4f}")

    targets = [float(t) / 100 for t in args.targets.split(",")]

    def topk_mask(gap, frac):
        valid = ~np.isnan(gap)
        vidx = np.flatnonzero(valid)
        order = vidx[np.argsort(-gap[vidx])]
        k = int(round(frac * len(order)))
        top = np.zeros(n, bool)
        top[order[:k]] = True
        return top

    rows = []
    for t in targets:
        router_take = []
        for tg in tags:
            tau, usage = solve_tau(d, tg, t)
            router_take.append(decide(d, tg, tau))
        router_take = np.array(router_take)  # (n_seeds, n_frames)
        realised_usage = router_take.mean()

        row = {"target": t * 100, "realised_usage": float(realised_usage) * 100}
        for name, gap in (("recall", rec_gap), ("precision", prec_gap)):
            top = topk_mask(gap, t)
            n_top = int(top.sum())
            router_hit = float(router_take[:, top].mean()) if n_top else float("nan")
            row[f"{name}_topk_n"] = n_top
            row[f"{name}_router_coverage"] = router_hit * 100
            row[f"{name}_random_coverage"] = realised_usage * 100
        rows.append(row)
        print(f"  target {t*100:>3.0f}% (realised {realised_usage*100:.1f}%): "
              f"recall-gap top-K router coverage {row['recall_router_coverage']:.1f}% "
              f"(random {row['recall_random_coverage']:.1f}%)  |  "
              f"precision-gap top-K router coverage {row['precision_router_coverage']:.1f}% "
              f"(random {row['precision_random_coverage']:.1f}%)")

    out = {"conf_thres": args.conf, "iou": 0.50, "n_frames": n,
           "n_frames_with_gt": int(has_gt.sum()), "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[*] {args.out}")


if __name__ == "__main__":
    main()
