"""Scene decomposition of base-better (A<0) frames on BDD100K.

A = L_base - L_super.  A<0 means SUPER hurts (base is the right choice).
Join the per-image advantage with BDD attribute labels (weather/scene/timeofday)
and geometry (object count, mean box area = object scale, crowding) to ask:
*what kind of scenes are base-better?*  Reports, for each attribute value, the
A<0 rate and mean advantage, plus a continuous-feature comparison A<0 vs A>0.

    python -m method02_advantage_regress_tinyConv.decompose_neg_adv
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

LBL = "/media/data/bdd100k_yolo/bdd100k_labels_images_val.json"


def load_attr(lbl):
    d = json.loads(Path(lbl).read_text())
    out = {}
    for r in d:
        a = r.get("attributes", {})
        boxes = [o["box2d"] for o in r.get("labels", []) if "box2d" in o]
        areas = [max(0.0, (b["x2"] - b["x1"])) * max(0.0, (b["y2"] - b["y1"])) for b in boxes]
        out[r["name"]] = {
            "weather": a.get("weather", "?"),
            "scene": a.get("scene", "?"),
            "timeofday": a.get("timeofday", "?"),
            "n_obj": len(boxes),
            "mean_area": float(np.mean(areas)) if areas else 0.0,
            "max_area": float(np.max(areas)) if areas else 0.0,
        }
    return out


def cat_table(name, vals, neg, A):
    """For one categorical attribute: per-value count, A<0 rate, mean A."""
    groups = defaultdict(list)
    for v, n, a in zip(vals, neg, A):
        groups[v].append((n, a))
    print(f"\n== {name} ==")
    print(f"{'value':<14}{'n':>6}{'%data':>7}{'A<0 rate':>10}{'mean A':>9}")
    base_rate = neg.mean()
    rows = sorted(groups.items(), key=lambda kv: -np.mean([x[0] for x in kv[1]]))
    for val, items in rows:
        ns = np.array([x[0] for x in items]); as_ = np.array([x[1] for x in items])
        rate = ns.mean()
        flag = "  <<" if rate > base_rate + 0.05 else ""
        print(f"{str(val):<14}{len(items):>6}{len(items)/len(vals)*100:>6.1f}%"
              f"{rate*100:>9.1f}%{as_.mean():>9.4f}{flag}")
    print(f"{'(overall)':<14}{len(vals):>6}{'100.0%':>7}{base_rate*100:>9.1f}%{A.mean():>9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="method02_advantage_regress_tinyConv/outputs/bdd100k/cache_val_g2.pt")
    ap.add_argument("--labels", default=LBL)
    args = ap.parse_args()

    c = torch.load(args.cache, map_location="cpu", weights_only=False)
    A = (c["loss_base"] - c["loss_super"]).numpy()
    stems = [Path(p).name for p in c["im_file"]]
    attr = load_attr(args.labels)
    miss = [s for s in stems if s not in attr]
    if miss:
        print(f"[!] {len(miss)} cache images missing in labels (dropped)")
    keep = np.array([s in attr for s in stems])
    A = A[keep]; stems = [s for s, k in zip(stems, keep) if k]
    neg = (A < 0).astype(float)
    print(f"N={len(A)}   A<0 (base-better)={neg.mean()*100:.1f}%   mean A={A.mean():.4f}")

    for field in ("weather", "scene", "timeofday"):
        cat_table(field, [attr[s][field] for s in stems], neg, A)

    # continuous geometry: compare A<0 vs A>0 distributions
    print("\n== geometry (A<0 base-better vs A>0 super-better) ==")
    print(f"{'feature':<12}{'A<0 median':>12}{'A>0 median':>12}{'A<0 mean':>11}{'A>0 mean':>11}")
    for feat in ("n_obj", "mean_area", "max_area"):
        x = np.array([attr[s][feat] for s in stems])
        xn, xp = x[A < 0], x[A > 0]
        print(f"{feat:<12}{np.median(xn):>12.1f}{np.median(xp):>12.1f}"
              f"{xn.mean():>11.1f}{xp.mean():>11.1f}")

    # advantage vs object scale buckets (is depth useless when objects are big/near?)
    area = np.array([attr[s]["mean_area"] for s in stems])
    q = np.quantile(area[area > 0], [0.25, 0.5, 0.75])
    print("\n== A<0 rate by mean object-area quartile ==")
    edges = [0] + list(q) + [area.max() + 1]
    names = ["Q1 small/far", "Q2", "Q3", "Q4 big/near"]
    for i, nm in enumerate(names):
        m = (area >= edges[i]) & (area < edges[i + 1])
        if m.sum():
            print(f"  {nm:<14} area[{edges[i]:.0f},{edges[i+1]:.0f})  "
                  f"n={m.sum():>4}  A<0 rate={neg[m].mean()*100:5.1f}%  mean A={A[m].mean():+.4f}")


if __name__ == "__main__":
    main()
