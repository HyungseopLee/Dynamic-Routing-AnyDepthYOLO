"""Plot feature-ablation Pareto with seed aggregation (mean +/- std).

Expects eval_video output where policy strategies are named
    policy_<feat>_s<seed>_t<tau>          e.g. policy_input_s3_t+020
(produced by passing --policies "input_s0=...,input_s1=...,...,both_s4=..." ).

For each feat (input/pred/both) it groups the 5 seeds by threshold tau, then
plots the per-tau mean curve (x = mean SUPER usage %, y = mean metric) with a
+/- std band, so overlapping bands => the feat difference is not significant.

Usage:
    python method01_advantage_regress/plot_ablation.py \
        --curve method01_advantage_regress/outputs/eval/video_curve_featabl_seeds.json \
        --out   method01_advantage_regress/outputs/eval/ablation_seeds
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FEAT_STYLE = {"input": ("tab:red", "backbone feat"),
              "pred":  ("tab:orange", "neck feat"),
              "both":  ("tab:brown", "both feat")}
NAME_RE = re.compile(r"policy_(input|pred|both)_s(\d+)_t([+-]\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="method01_advantage_regress/outputs/eval/video_curve_featabl_seeds.json")
    ap.add_argument("--out", default="method01_advantage_regress/outputs/eval/ablation_seeds")
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    args = ap.parse_args()

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]
    gb, gs = data["gflops_base"], data["gflops_super"]

    # feat -> tau -> list over seeds of (super_rate, metric)
    agg = defaultdict(lambda: defaultdict(list))
    endpoints = {}
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            endpoints[r["name"]] = (r["super_rate"] * 100, r[args.metric])
        m = NAME_RE.match(r["name"])
        if not m:
            continue
        feat, _seed, tau = m.group(1), m.group(2), m.group(3)
        agg[feat][tau].append((r["super_rate"] * 100, r[args.metric]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for feat, taus in agg.items():
        color, lbl = FEAT_STYLE[feat]
        pts = []
        for tau, seeds in taus.items():
            arr = np.array(seeds)                      # [n_seed, 2]
            x = arr[:, 0].mean()
            y_mean = arr[:, 1].mean(); y_std = arr[:, 1].std()
            pts.append((x, y_mean, y_std, len(seeds)))
        pts.sort()
        xs = [p[0] for p in pts]; ym = [p[1] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ym, "o-", color=color, label=f"{lbl} (n={pts[0][3]})", markersize=4)
        ax.fill_between(xs, np.array(ym) - np.array(ys), np.array(ym) + np.array(ys),
                        color=color, alpha=0.18)

    for nm, (x, y) in endpoints.items():
        ax.scatter([x], [y], marker="*", s=220, zorder=5,
                   color="black" if nm == "always_super" else "darkgreen")
        ax.annotate(nm.replace("always_", ""), (x, y), textcoords="offset points", xytext=(6, 6))
    if "always_super" in endpoints:
        ax.axhline(endpoints["always_super"][1], color="black", ls=":", alpha=0.4)

    ax.set_xlabel("SUPER usage (%)"); ax.set_ylabel(args.metric.upper())
    s2g = lambda sp: (sp / 100.0) * gs + (1 - sp / 100.0) * gb
    g2s = lambda g: (g - gb) / (gs - gb) * 100.0
    sec = ax.secondary_xaxis("top", functions=(s2g, g2s)); sec.set_xlabel("GFLOPs (per frame)")
    ax.set_title("Feature ablation (mean +/- std over seeds)", pad=28)
    ax.legend(); ax.grid(alpha=0.3)
    out = f"{args.out}_{args.metric}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
