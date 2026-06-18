"""Main-result video routing curve (single learned-policy family), in the same
format as the KITTI / BDD100K main figures:

  - x-axis      = SUPER usage (%)   (0% = always-base, 100% = always-super)
  - top x-axis  = GFLOPs / frame    (exact linear function of super usage)
  - y-axis      = AP50 or AP50:95

The eval names policies `policy_seed{0..4}_b{budget}` (one tinyConv config swept
over FLOPs budgets with 5 seeds); PI rows (`_pi{budget}`) are excluded here and
plotted separately. We draw the policy mean +/- std band over seeds and overlay
the no-train routing baselines (random / luminance / edge / confidence) plus the
always-base/super anchors, spanning each curve to the anchors for a full Pareto.

Usage:
    python method02_advantage_regress_tinyConv/plot_video_curve_main.py \
        --curve .../eval_both/video_curve.json --out .../eval_both/fig_main \
        --metric map50 --title "Waymo Open (FRONT) feat=both: learned policy vs baselines"
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
POLICY_RE = re.compile(r"^policy_seed(\d+)_b(\d+)$")   # budget sweep only (skips _pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--out", required=True, help="output path stem (_<metric>.png appended)")
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    ap.add_argument("--title", default="Video depth routing: learned policy vs no-train baselines")
    args = ap.parse_args()

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]

    bases = defaultdict(list)                 # family -> [(super%, metric)]
    pol = defaultdict(list)                   # budget -> [(super%, metric) over seeds]
    anchors = {}
    g_base = g_super = None
    for r in rows:
        x = r["super_rate"] * 100.0
        if r["name"] == "always_base":  g_base = r["gflops"]
        if r["name"] == "always_super": g_super = r["gflops"]
        if r["name"] in ("always_base", "always_super"):
            anchors[r["name"]] = (x, r[args.metric]); continue
        m = POLICY_RE.match(r["name"])
        if m:
            pol[int(m.group(2))].append((x, r[args.metric])); continue
        fam = r.get("family") or r.get("kind")
        if fam in BASE_STYLE:
            bases[fam].append((x, r[args.metric]))

    fig, ax = plt.subplots(figsize=(9, 6))

    # 0% super == always_base, 100% == always_super -> span every curve to the
    # anchors for a complete base->super Pareto front.
    ab = anchors.get("always_base"); asu = anchors.get("always_super")

    def span(xs, ys):
        if ab is not None: xs = [ab[0]] + list(xs); ys = [ab[1]] + list(ys)
        if asu is not None: xs = list(xs) + [asu[0]]; ys = list(ys) + [asu[1]]
        return xs, ys

    for fam, pts in bases.items():
        pts = sorted(pts)
        fmt, color, lbl = BASE_STYLE[fam]
        xs, ys = span([p[0] for p in pts], [p[1] for p in pts])
        ax.plot(xs, ys, fmt, color=color, label=lbl, markersize=4,
                linewidth=1.2, alpha=0.8)

    # learned policy: mean +/- std over seeds, per budget
    pts = []
    for b, seeds in pol.items():
        arr = np.array(seeds)                 # [n_seed, 2]
        pts.append((arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 1].std(), len(seeds), b))
    pts.sort()
    xs = [p[0] for p in pts]; ym = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
    nseed = pts[0][3] if pts else 0
    xs_s, ym_s = span(xs, ym)
    # std band also spans to the anchors; the always_base/super endpoints are
    # single-eval points with no seed variance, so std=0 there (band tapers in).
    xs_b, lo_b = span(xs, ym - ys); _, hi_b = span(xs, ym + ys)
    ax.plot(xs_s, ym_s, "-", color="tab:red", linewidth=1.8, zorder=3)
    ax.plot(xs, ym, "o", color="tab:red", label=f"learned policy (n={nseed})",
            markersize=5, zorder=4)
    ax.fill_between(xs_b, lo_b, hi_b, color="tab:red", alpha=0.18)
    for x, y, _s, _n, b in pts:
        ax.annotate(f"{b}%", (x, y), textcoords="offset points", xytext=(0, 6),
                    fontsize=7, color="tab:red", ha="center")

    for nm, (x, y) in anchors.items():
        ax.scatter([x], [y], marker="*", s=240, zorder=5,
                   color="black" if nm == "always_super" else "darkgreen")
        ax.annotate(nm.replace("always_", ""), (x, y),
                    textcoords="offset points", xytext=(6, 6))
    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="black", ls=":", alpha=0.4)

    ax.set_xlabel("SUPER usage (%)")
    ax.set_ylabel(args.metric.upper())
    ax.set_title(args.title, pad=24)
    ax.legend(); ax.grid(alpha=0.3)

    # top axis: GFLOPs/frame = g_base + (super%/100)*(g_super - g_base)
    if g_base is not None and g_super is not None:
        pct2gf = lambda p: g_base + (p / 100.0) * (g_super - g_base)
        gf2pct = lambda g: (g - g_base) / (g_super - g_base) * 100.0
        secax = ax.secondary_xaxis("top", functions=(pct2gf, gf2pct))
        secax.set_xlabel("GFLOPs (per frame)")

    out = f"{args.out}_{ {'map50': 'ap50', 'map': 'ap5095'}[args.metric] }.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
