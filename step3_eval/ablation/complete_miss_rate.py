"""Complete-miss rate for safety-critical classes (Comment 3): global mAP/AR
averages can conceal frames where a safety-critical class is present in the GT
but the detector returns zero true positives for it -- a "complete miss," the
literal single-frame failure the reviewer's comment names.

For pedestrian and cyclist separately, and restricted to frames where that
class's GT count > 0, we compute the fraction of frames with zero TP (at a
fixed conf=0.25 operating point, IoU=0.50) under always-base, always-super,
and the router at several target SUPER budgets.

Usage (run from repo root):
    python -m step3_eval.ablation.complete_miss_rate \
        --dump results/step3_eval/kitti/eval/frames.npz \
        --out results/step3_eval/kitti/eval/complete_miss_rate.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from step3_eval.ablation.tau_sweep import load, decide, solve_tau

CONF = 0.25
IOU_IDX = 0  # IoU=0.50
SAFETY = {"pedestrian": 3, "cyclist": 5}


def per_frame_tp_gt(d, path, cls_id, conf_thres=CONF):
    """Per-frame (tp_count, gt_count) for one class, at a fixed confidence."""
    off = d[f"off_{path}"]
    conf = d[f"conf_{path}"]
    cls = d[f"cls_{path}"]
    tp = d[f"tp_{path}"][:, IOU_IDX]
    gt_col = int(np.flatnonzero(d["gt_cls"] == cls_id)[0])
    gt_per_frame = d["gt_count"][:, gt_col].astype(np.int64)

    n = len(off) - 1
    tp_count = np.zeros(n, np.int64)
    for i in range(n):
        s, e = off[i], off[i + 1]
        keep = (conf[s:e] >= conf_thres) & (cls[s:e] == cls_id)
        tp_count[i] = int(tp[s:e][keep].sum())
    return tp_count, gt_per_frame


def miss_rate(tp_count, gt_count, subset=None):
    has_gt = gt_count > 0
    if subset is not None:
        has_gt = has_gt & subset
    n = int(has_gt.sum())
    if n == 0:
        return float("nan"), 0
    missed = (tp_count[has_gt] == 0).sum()
    return float(missed) / n * 100, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--targets", default="10,30,50,70,90")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load(args.dump)
    tags = d["tags"]
    targets = [float(t) / 100 for t in args.targets.split(",")]

    out = {"conf_thres": args.conf, "iou": 0.50, "classes": {}}

    for cname, cid in SAFETY.items():
        tp_b, gt = per_frame_tp_gt(d, "base", cid, args.conf)
        tp_s, _ = per_frame_tp_gt(d, "super", cid, args.conf)

        base_rate, n_base = miss_rate(tp_b, gt)
        super_rate, n_super = miss_rate(tp_s, gt)
        print(f"[{cname}] frames with GT>0: {n_base}")
        print(f"  always-base : complete-miss rate = {base_rate:.2f}%")
        print(f"  always-super: complete-miss rate = {super_rate:.2f}%")

        rows = []
        for t in targets:
            router_rates = []
            for tg in tags:
                tau, _usage = solve_tau(d, tg, t)
                take = decide(d, tg, tau)  # True where SUPER is chosen
                tp_router = np.where(take, tp_s, tp_b)
                r, _ = miss_rate(tp_router, gt)
                router_rates.append(r)
            r_mean = float(np.mean(router_rates))
            r_std = float(np.std(router_rates))
            rows.append({"target": t * 100, "router_miss_rate": r_mean,
                        "router_miss_rate_std": r_std})
            print(f"  router @ {t*100:>3.0f}% budget: complete-miss rate = "
                  f"{r_mean:.2f}% (std {r_std:.2f})")

        out["classes"][cname] = {
            "n_frames_with_gt": n_base,
            "always_base_miss_rate": base_rate,
            "always_super_miss_rate": super_rate,
            "rows": rows,
        }
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[*] {args.out}")


if __name__ == "__main__":
    main()
