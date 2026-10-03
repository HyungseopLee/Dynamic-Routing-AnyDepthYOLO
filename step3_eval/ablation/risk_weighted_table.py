"""Risk-weighted loss ablation on KITTI: safety-critical vs other classes.

Rows are the risk weight w in the router's regression target

    L_risk(w) = L(all GT) + (w - 1) * L(safety-critical GT only),

so w = 1 is the paper's plain advantage (no safety weighting) and w > 1 up-weights
pedestrian / person_sitting / cyclist. Columns split the classes the same way, so
the table shows whether weighting the loss towards those classes actually buys
anything on them -- and what it costs elsewhere.

Each router is evaluated at 9 budgets x 5 seeds; the seeds are averaged per budget
and the curves are interpolated onto a common SUPER-rate grid, so every row is read
at the same compute budget. always_base / always_super bound what any routing
policy can reach, since the detector is frozen.

Usage (run from repo root):
    python -m step3_eval.ablation.risk_weighted_table
    python -m step3_eval.ablation.risk_weighted_table --pattern video_curve_coco_riskw{w}.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

EVAL_DIR = Path("results/step3_eval/kitti/eval")
NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian", 4: "person_sitting",
         5: "cyclist", 6: "tram"}
SAFETY = [3, 4, 5]                      # pedestrian, person_sitting, cyclist
GRID = np.arange(0.20, 0.95, 0.05)      # common SUPER-rate grid
WEIGHTS = [1, 2, 5, 10]


def group_metric(row, cls_ids, key):
    """Mean of `key` over the given classes, skipping ones absent from the GT."""
    vals = [row["per_class"][str(c)][key] for c in cls_ids if str(c) in row["per_class"]]
    return float(np.mean(vals)) if vals else np.nan


def seed_curve(seeds, fn):
    """Seed-averaged fn(row) interpolated onto GRID."""
    out = []
    for rows in seeds.values():
        rows = sorted(rows, key=lambda r: r["super_rate"])
        out.append(np.interp(GRID, [r["super_rate"] for r in rows],
                             [fn(r) for r in rows], left=np.nan, right=np.nan))
    return np.nanmean(np.array(out), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="video_curve_riskw{w}.json",
                    help="eval json name per weight, with {w} for the risk weight")
    ap.add_argument("--dir", default=str(EVAL_DIR))
    args = ap.parse_args()
    d = Path(args.dir)

    data, consts = {}, None
    for w in WEIGHTS:
        p = d / args.pattern.format(w=w)
        if not p.exists():
            raise SystemExit(f"missing {p}")
        rows = json.loads(p.read_text())["rows"]
        seeds = {}
        for r in rows:
            if r["kind"] == "router":
                seeds.setdefault(r["family"], []).append(r)
        data[w] = seeds
        if consts is None:
            consts = {r["name"]: r for r in rows if r["kind"] == "const"}

    present = sorted({int(c) for r in consts.values() for c in r["per_class"]})
    sc = [c for c in present if c in SAFETY]
    other = [c for c in present if c not in SAFETY]
    print(f"safety-critical : {', '.join(NAMES[c] for c in sc)}")
    print(f"other           : {', '.join(NAMES[c] for c in other)}")
    print(f"read at equal budget, mean over SUPER rate {GRID[0]*100:.0f}-{GRID[-1]*100:.0f}%, "
          f"5 seeds\n")

    cols = [("safety-critical AP", sc, "ap"), ("safety-critical AR", sc, "ar"),
            ("other AP", other, "ap"), ("other AR", other, "ar")]
    w_col = 21
    print(f"{'router':<22}" + "".join(f"{c[0]:>21}" for c in cols))
    print("-" * (22 + w_col * len(cols)))

    for name, tag in (("always_base", "always BASE (lower)"),
                      ("always_super", "always SUPER (upper)")):
        r = consts.get(name)
        if r is None:
            continue
        print(f"{tag:<22}" + "".join(f"{group_metric(r, ids, k):>21.4f}"
                                     for _, ids, k in cols))
    print("-" * (22 + w_col * len(cols)))

    base_row = None
    for w in WEIGHTS:
        label = "w=1  (no weighting)" if w == 1 else f"w={w}"
        vals = [np.nanmean(seed_curve(data[w], lambda r, i=ids, k=k: group_metric(r, i, k)))
                for _, ids, k in cols]
        if base_row is None:
            base_row = vals
            print(f"{label:<22}" + "".join(f"{v:>21.4f}" for v in vals))
        else:
            print(f"{label:<22}" + "".join(
                f"{v:>13.4f} ({v-b:+.4f})" for v, b in zip(vals, base_row)))
    print("\n(parentheses: change vs the unweighted w=1 router at the same budget)")


if __name__ == "__main__":
    main()
