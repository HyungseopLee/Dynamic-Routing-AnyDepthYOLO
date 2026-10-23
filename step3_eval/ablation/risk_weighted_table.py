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
WEIGHTS = [1, 2, 3, 4, 5, 10]


def group_metric(row, cls_ids, key):
    """Mean of `key` over the given classes, skipping ones absent from the GT."""
    vals = [row["per_class"][str(c)][key] for c in cls_ids if str(c) in row["per_class"]]
    return float(np.mean(vals)) if vals else np.nan


def seed_curve(seeds, fn):
    """Seed-averaged fn(row) interpolated onto GRID."""
    return np.nanmean(seed_values(seeds, fn, GRID), axis=0)


def seed_values(seeds, fn, targets):
    """fn(row) per seed, interpolated onto `targets` on the SUPER-rate axis.

    Each seed's own curve is interpolated separately, so a row of the table is read
    at the same realised SUPER usage for every seed and every risk weight. Returning
    the per-seed values rather than their mean is what lets the caller report a
    spread: the differences between risk weights here are a few tenths of an AP
    point, which is not interpretable without one.
    """
    out = []
    for rows in seeds.values():
        rows = sorted(rows, key=lambda r: r["super_rate"])
        out.append(np.interp(targets, [r["super_rate"] for r in rows],
                             [fn(r) for r in rows], left=np.nan, right=np.nan))
    return np.array(out)



def oracle_rows(dump, sc, other, targets):
    """Upper bound on what ANY router could reach for the safety-critical classes.

    The router only ever picks between two executions of the same frozen detector, so
    for every frame there is a best choice, and a policy with perfect foresight would
    make it. We approximate that choice on the metric itself rather than on the
    training target: per frame, the number of safety-critical ground-truth boxes
    recovered as true positives (IoU 0.50, conf 0.25) under each path. Frames are
    ranked by super minus base on that count and the top `target` fraction is routed
    to SUPER, so the bound is read at the same budget as the learned rows. The
    unconstrained bound takes every frame with a positive gain, ignoring the budget.
    """
    from step3_eval.ablation.tau_sweep import load as load_dump, score
    d = load_dump(dump)
    n = len(d["seq"])
    gt_cols = {int(c): i for i, c in enumerate(d["gt_cls"])}
    gain = np.zeros(n)
    for pth, sign in (("super", 1.0), ("base", -1.0)):
        off, cls, conf = d[f"off_{pth}"], d[f"cls_{pth}"], d[f"conf_{pth}"]
        tp = d[f"tp_{pth}"][:, 0]
        keep = (conf >= 0.25) & np.isin(cls, sc) & tp
        idx = np.flatnonzero(keep)
        per_frame = np.zeros(n)
        # map each kept detection row back to its frame
        frame_of = np.searchsorted(off, idx, side="right") - 1
        np.add.at(per_frame, frame_of, 1.0)
        gain += sign * per_frame
    order = np.argsort(-gain, kind="stable")
    out = []
    for t in list(targets) + [None]:
        take = np.zeros(n, bool)
        if t is None:
            take[gain > 0] = True
            label = "oracle (unconstrained)"
        else:
            take[order[: int(round(t * n))]] = True
            label = f"oracle @ {t*100:.0f}%"
        pc = score(d, take, per_class=True)[3]
        vals = [np.mean([pc[c][k] for c in ids if c in pc]) * 100
                for ids, k in ((sc, "ap"), (sc, "ar"), (other, "ap"), (other, "ar"))]
        out.append((label, take.mean() * 100, vals))
    return out


def report_targets(data, consts, sc, other, targets, dump=None):
    """Table read at fixed SUPER usages, with the spread over the 5 seeds."""
    cols = [("sc AP", sc, "ap"), ("sc AR", sc, "ar"),
            ("other AP", other, "ap"), ("other AR", other, "ar")]
    for t in targets:
        print(f"\n=== SUPER usage {t*100:.0f}% (mean +/- std over 5 seeds) ===")
        print(f"{'router':<24}" + "".join(f"{c[0]:>18}" for c in cols))
        for name, tag in (("always_base", "always-base"),
                          ("always_super", "always-super")):
            r = consts.get(name)
            if r is not None:
                print(f"{tag:<24}" + "".join(
                    f"{group_metric(r, ids, k)*100:>18.2f}" for _, ids, k in cols))
        for w in WEIGHTS:
            cells = []
            for _, ids, k in cols:
                v = seed_values(data[w],
                                lambda r, i=ids, kk=k: group_metric(r, i, kk), [t])[:, 0]
                cells.append(f"{np.nanmean(v)*100:>11.2f} +/-{np.nanstd(v)*100:.2f}")
            print(f"{'w='+str(w):<24}" + "".join(f"{c:>18}" for c in cells))
        if dump:
            for label, used, vals in oracle_rows(dump, sc, other, [t]):
                if "unconstrained" in label:
                    continue
                print(f"{label+f' [{used:.1f}%]':<24}"
                      + "".join(f"{v:>18.2f}" for v in vals))
    if dump:
        print("\n=== budget-free upper bound ===")
        for label, used, vals in oracle_rows(dump, sc, other, []):
            print(f"{label+f' [{used:.1f}% SUPER]':<30}"
                  + "".join(f"{v:>18.2f}" for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="video_curve_riskw{w}.json",
                    help="eval json name per weight, with {w} for the risk weight")
    ap.add_argument("--dir", default=str(EVAL_DIR))
    ap.add_argument("--targets", default=None,
                    help="comma-separated SUPER usages in percent; reads the table at "
                         "each of them with a seed spread instead of averaging the curve")
    ap.add_argument("--dump", default=None,
                    help="frame dump, to add the oracle upper bound rows")
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

    if args.targets:
        report_targets(data, consts, sc, other,
                       [float(x) / 100 for x in args.targets.split(",")], args.dump)
        return

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
