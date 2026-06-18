"""Main-result Pareto figure (paper style): AP vs SUPER usage (bottom x) with a twin
GFLOPs/frame top axis. The learned policy is drawn as a mean+/-std band over seeds;
no-train baselines (random, luminance, edge density, confidence) are dashed lines.

Reproduces fig_main_*_ap5095.pdf with the learned policy = the chosen feature source
(default: the capacity-strengthened both-feature router) and the confidence baseline
included.

    python -m method02_advantage_regress_tinyConv.make_main_figure \
        --curve outputs/bdd100k/eval/video_curve_archabl.json \
        --policy_fam 'policy_bn(\\d+)' --out ../paper/_extracted/fig_main_bdd_ap5095.pdf
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# baseline family -> (color, marker, linestyle, label)
BASE_STYLE = {
    "random":    ("tab:gray",   "x", (0, (6, 3)),  "random"),
    "lum":       ("tab:purple", "v", (0, (6, 3)),  "luminance"),
    "edge":      ("tab:green",  "P", (0, (6, 3)),  "edge density"),
    "conftop20": ("tab:orange", "s", (0, (6, 3)),  "confidence"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--policy_fam", default=r"policy_bn(\d+)",
                    help="regex matching the policy family; group(1) = seed id")
    ap.add_argument("--metric", default="map", choices=["map", "map50"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]
    gb, gs = data["gflops_base"], data["gflops_super"]
    pol_re = re.compile(args.policy_fam)
    AP = lambda r: r[args.metric] * 100.0   # AP in percent (e.g. 22.4, not 0.224)

    bases = defaultdict(list)        # family -> [(super_rate, metric)]
    pol = defaultdict(list)          # sweep-key -> [(super_rate, metric) over seeds]
    anchors = {}
    for r in rows:
        nm = r["name"]
        if nm in ("always_base", "always_super"):
            anchors[nm] = (r["super_rate"], AP(r)); continue
        if pol_re.fullmatch(r.get("family", "")):
            key = r["budget"] if r.get("budget") is not None else r["thres"]
            pol[key].append((r["super_rate"], AP(r))); continue
        fam = r.get("family")
        if fam in BASE_STYLE:
            bases[fam].append((r["super_rate"], AP(r)))

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linestyle": "--"})
    fig, ax = plt.subplots(figsize=(5.0, 3.7))

    # baselines (under the policy)
    for fam in ("random", "lum", "edge", "conftop20"):
        if fam not in bases:
            continue
        pts = sorted(bases[fam])
        c, mk, ls, lbl = BASE_STYLE[fam]
        ax.plot([p[0] * 100 for p in pts], [p[1] for p in pts], marker=mk, color=c,
                ls=ls, label=lbl, markersize=4, linewidth=1.3, alpha=0.85)

    # learned policy: mean +/- std over seeds, sorted by mean super usage
    pts = []
    for key, seeds in pol.items():
        a = np.array(seeds)
        pts.append((a[:, 0].mean() * 100, a[:, 1].mean(), a[:, 1].std(), len(seeds)))
    pts.sort()
    nseed = pts[0][3] if pts else 0
    # The router's tau->+-inf extremes ARE the always-base / always-super anchors, so
    # extend the policy curve to span the full 0-100% SUPER-usage range (std=0 there).
    if "always_base" in anchors:
        pts.insert(0, (0.0, anchors["always_base"][1], 0.0, nseed))
    if "always_super" in anchors:
        pts.append((100.0, anchors["always_super"][1], 0.0, nseed))
    xs = np.array([p[0] for p in pts]); ym = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    ax.fill_between(xs, ym - ys, ym + ys, color="tab:red", alpha=0.18, zorder=3)
    ax.plot(xs, ym, "o-", color="tab:red", label="ours", markersize=4.5,
            linewidth=1.9, zorder=4)

    # anchors: endpoints + dotted super ceiling
    for nm, (x, y) in anchors.items():
        ax.scatter([x * 100], [y], marker="*", s=90, zorder=5, color="black")
    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="0.4", ls=":", lw=1, alpha=0.6)

    ax.set_xlabel("SUPER usage (\\%)" if matplotlib.rcParams["text.usetex"]
                  else "SUPER usage (%)")
    ax.set_ylabel(r"AP$_{50:95}$")
    ax.set_xlim(-3, 103)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # twin top axis: GFLOPs/frame (linear in SUPER usage)
    def s2g(s):  # super% -> gflops
        return gb + (s / 100.0) * (gs - gb)

    def g2s(g):
        return (g - gb) / (gs - gb) * 100.0
    sec = ax.secondary_xaxis("top", functions=(s2g, g2s))
    sec.set_xlabel("GFLOPs / frame")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print(f"[*] policy seeds={nseed}, points={len(pts)} -> {args.out}")


if __name__ == "__main__":
    main()
