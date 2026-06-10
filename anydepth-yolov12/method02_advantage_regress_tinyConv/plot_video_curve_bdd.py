"""Plot the BDD100K MOT depth-routing curve produced by merge_video_shards.py.

Unlike plot_ablation.py (KITTI feature-ablation, grouped by `input_s{seed}_t{tau}`),
the BDD eval names policies `policy_seed{0..4}_b{budget}`: a single tinyConv config
swept over FLOPs *budgets* with 5 seeds. So we group the policy by budget, draw the
mean +/- std band over seeds, and overlay the no-train routing baselines
(random / luminance / edge / confidence) plus the always-base/super anchors.

Usage:
    python method02_advantage_regress_tinyConv/plot_video_curve_bdd.py \
        --curve method02_advantage_regress_tinyConv/outputs/bdd100k/eval/video_curve.json \
        --out   method02_advantage_regress_tinyConv/outputs/bdd100k/eval/fig_main \
        --metric map50
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

BASE_STYLE = {                       # family -> (fmt, color, label)
    "random":    ("x--", "tab:gray",   "random"),
    "lum":       ("s--", "tab:green",  "luminance"),
    "edge":      ("D--", "tab:orange", "edge density"),
    "conftop20": ("^--", "tab:blue",   "confidence (top-20)"),
}
POLICY_RE = re.compile(r"^policy_seed(\d+)_b(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--out", required=True, help="output path stem (_<metric>.png appended)")
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    args = ap.parse_args()

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]
    gb, gs = data["gflops_base"], data["gflops_super"]

    bases = defaultdict(list)                 # family -> [(gflops, metric)]
    pol = defaultdict(list)                   # budget -> [(gflops, metric) over seeds]
    anchors = {}
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            anchors[r["name"]] = (r["gflops"], r[args.metric]); continue
        m = POLICY_RE.match(r["name"])
        if m:
            pol[int(m.group(2))].append((r["gflops"], r[args.metric])); continue
        fam = r.get("family") or r.get("kind")
        if fam in BASE_STYLE:
            bases[fam].append((r["gflops"], r[args.metric]))

    fig, ax = plt.subplots(figsize=(9, 6))

    # baselines (under the policy curve)
    for fam, pts in bases.items():
        pts = sorted(pts)
        fmt, color, lbl = BASE_STYLE[fam]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], fmt, color=color,
                label=lbl, markersize=4, linewidth=1.2, alpha=0.8)

    # learned policy: mean +/- std over seeds, per budget
    pts = []
    for b, seeds in pol.items():
        arr = np.array(seeds)                 # [n_seed, 2]
        pts.append((arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 1].std(), len(seeds), b))
    pts.sort()
    xs = [p[0] for p in pts]; ym = [p[1] for p in pts]; ys = [p[2] for p in pts]
    nseed = pts[0][3] if pts else 0
    ax.plot(xs, ym, "o-", color="tab:red", label=f"learned policy (n={nseed})",
            markersize=5, linewidth=1.8, zorder=4)
    ax.fill_between(xs, np.array(ym) - np.array(ys), np.array(ym) + np.array(ys),
                    color="tab:red", alpha=0.18)
    for x, y, _s, _n, b in pts:
        ax.annotate(f"{b}%", (x, y), textcoords="offset points", xytext=(0, 6),
                    fontsize=7, color="tab:red", ha="center")

    # anchors
    for nm, (x, y) in anchors.items():
        ax.scatter([x], [y], marker="*", s=240, zorder=5,
                   color="black" if nm == "always_super" else "darkgreen")
        ax.annotate(nm.replace("always_", ""), (x, y),
                    textcoords="offset points", xytext=(6, 6))
    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="black", ls=":", alpha=0.4)

    ax.set_xlabel("GFLOPs (per frame)")
    ax.set_ylabel(args.metric.upper())
    ax.set_title("BDD100K MOT depth routing: learned policy vs no-train baselines", pad=12)
    ax.legend(); ax.grid(alpha=0.3)
    out = f"{args.out}_{ {'map50': 'ap50', 'map': 'ap5095'}[args.metric] }.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
