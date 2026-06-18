"""Paper-style main Pareto figure (matches fig_main_ap5095 / fig_main_bdd_ap5095).

Clean, title-less single-panel figure for the paper main result:
  - x-axis      = SUPER usage (%)        (0 % = always-base, 100 % = always-super)
  - top x-axis  = GFLOPs / frame         (exact linear function of super usage)
  - y-axis      = AP_{50:95} (or AP_{50})
  - red "ours" curve = learned advantage-regression router (mean +/- std, 5 seeds)
  - baselines   = random / luminance / edge density / confidence
  - black stars at the always-base / always-super endpoints

Colors/markers are fixed to match the existing KITTI/BDD panels so the three
sub-figures (a)/(b)/(c) read consistently. Emits a PDF.

Usage:
    python method02_advantage_regress_tinyConv/plot_paper_main.py \
        --curve .../eval_both/video_curve.json \
        --out   .../fig_main_waymo_ap5095.pdf --metric map
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

# family -> (color, marker, label) -- fixed to match the paper KITTI/BDD panels
BASE_STYLE = {
    "random":    ("tab:gray",   "x", "random"),
    "lum":       ("tab:purple", "v", "luminance"),
    "edge":      ("tab:green",  "P", "edge density"),
    "conftop20": ("tab:orange", "s", "confidence"),
}
POLICY_RE = re.compile(r"^policy_seed(\d+)_b(\d+)$")   # budget sweep (skips _pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--out", required=True, help="output PDF path")
    ap.add_argument("--metric", default="map", choices=["map50", "map"])
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 13,
    })

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]
    AP = lambda r: r[args.metric] * 100.0   # AP reported in percent (e.g. 22.4, not 0.224)

    bases = defaultdict(list)              # family -> [(super%, metric)]
    pol = defaultdict(list)                # budget -> [(super%, metric) over seeds]
    anchors = {}
    g_base = g_super = None
    for r in rows:
        x = r["super_rate"] * 100.0
        if r["name"] == "always_base":  g_base = r["gflops"]
        if r["name"] == "always_super": g_super = r["gflops"]
        if r["name"] in ("always_base", "always_super"):
            anchors[r["name"]] = (x, AP(r)); continue
        m = POLICY_RE.match(r["name"])
        if m:
            pol[int(m.group(2))].append((x, AP(r))); continue
        fam = r.get("family") or r.get("kind")
        if fam in BASE_STYLE:
            bases[fam].append((x, AP(r)))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    ab = anchors.get("always_base"); asu = anchors.get("always_super")

    def span(xs, ys):              # anchor each curve to the base/super endpoints
        if ab is not None: xs = [ab[0]] + list(xs); ys = [ab[1]] + list(ys)
        if asu is not None: xs = list(xs) + [asu[0]]; ys = list(ys) + [asu[1]]
        return xs, ys

    for fam in ("random", "lum", "edge", "conftop20"):
        if fam not in bases:
            continue
        pts = sorted(bases[fam])
        color, mk, lbl = BASE_STYLE[fam]
        xs, ys = span([p[0] for p in pts], [p[1] for p in pts])
        ax.plot(xs, ys, linestyle="--", marker=mk, color=color, label=lbl,
                markersize=5, linewidth=1.4, alpha=0.85)

    # learned policy: mean +/- std over seeds, per budget
    pts = []
    for b, seeds in pol.items():
        arr = np.array(seeds)
        pts.append((arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 1].std()))
    pts.sort()
    xs = [p[0] for p in pts]; ym = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
    xs_s, ym_s = span(xs, ym)
    xs_b, lo_b = span(xs, ym - ys); _, hi_b = span(xs, ym + ys)
    ax.fill_between(xs_b, lo_b, hi_b, color="tab:red", alpha=0.20, zorder=2)
    ax.plot(xs_s, ym_s, "-", color="tab:red", linewidth=2.4, zorder=3)
    ax.plot(xs, ym, "o", color="tab:red", label="ours", markersize=6, zorder=4)

    # endpoints
    for nm, (x, y) in anchors.items():
        ax.scatter([x], [y], marker="*", s=320, color="black", zorder=6,
                   edgecolors="black")
    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="black", ls=":", alpha=0.5,
                   linewidth=1.0)

    ax.set_xlabel("SUPER usage (%)")
    ylab = r"AP$_{50:95}$" if args.metric == "map" else r"AP$_{50}$"
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.9)

    if g_base is not None and g_super is not None:
        pct2gf = lambda p: g_base + (p / 100.0) * (g_super - g_base)
        gf2pct = lambda g: (g - g_base) / (g_super - g_base) * 100.0
        secax = ax.secondary_xaxis("top", functions=(pct2gf, gf2pct))
        secax.set_xlabel("GFLOPs / frame")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, facecolor="white", edgecolor="none")
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
