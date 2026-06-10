"""Overlay two learned-policy families on the BDD100K MOT routing curve.

Same as plot_video_curve_bdd.py but the eval ran *two* tinyConv families together:
  bb{seed} = feat=input  (backbone-only router)
  bn{seed} = feat=both   (backbone + neck/pred router, capacity-strengthened)
Each is swept over FLOPs budgets with 5 seeds, so we draw a mean +/- std band per
family and overlay the no-train baselines + always-base/super anchors.

Usage:
    python method02_advantage_regress_tinyConv/plot_video_curve_both.py \
        --curve .../eval/video_curve_both20.json --out .../eval/fig_both20 --metric map50
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

BASE_STYLE = {
    "random":    ("x--", "tab:gray",   "random"),
    "lum":       ("s--", "tab:green",  "luminance"),
    "edge":      ("D--", "tab:orange", "edge density"),
    "conftop20": ("^--", "tab:blue",   "confidence (top-20)"),
}
POLICY_RE = re.compile(r"^policy_(bb|bn|pn)(\d+)_b(\d+)$")
FAM_STYLE = {                            # family -> (color, label)
    "bb": ("tab:red",    "router: backbone-only"),
    "bn": ("tab:purple", "router: backbone+neck"),
    "pn": ("tab:brown",  "router: neck-only"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--out", required=True, help="output path stem (_<metric>.png appended)")
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    ap.add_argument("--title", default="BDD100K MOT depth routing: backbone / backbone+neck / neck router")
    args = ap.parse_args()

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]

    # x-axis = SUPER usage (%); the top axis shows GFLOPs/frame, which is an exact
    # linear function of super-rate (base path at 0 %, super path at 100 %).
    bases = defaultdict(list)                          # family -> [(super%, metric)]
    pol = {fam: defaultdict(list) for fam in FAM_STYLE}        # fam -> budget -> [(super%, metric)]
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
            pol[m.group(1)][int(m.group(3))].append((x, r[args.metric])); continue
        fam = r.get("family") or r.get("kind")
        if fam in BASE_STYLE:
            bases[fam].append((x, r[args.metric]))

    fig, ax = plt.subplots(figsize=(9, 6))

    # Every routing curve spans the full base->super range: a 0% super-budget is
    # exactly always_base and 100% is exactly always_super, so anchor each curve at
    # those endpoints to draw a complete Pareto front (no extra eval needed).
    ab = anchors.get("always_base"); asu = anchors.get("always_super")

    def span(xs, ys):
        if ab is not None: xs = [ab[0]] + list(xs); ys = [ab[1]] + list(ys)
        if asu is not None: xs = list(xs) + [asu[0]]; ys = list(ys) + [asu[1]]
        return xs, ys

    for fam, pts in bases.items():
        pts = sorted(pts)
        fmt, color, lbl = BASE_STYLE[fam]
        xs, ys = span([p[0] for p in pts], [p[1] for p in pts])
        ax.plot(xs, ys, fmt, color=color,
                label=lbl, markersize=4, linewidth=1.2, alpha=0.8)

    for fam in FAM_STYLE:
        color, lbl = FAM_STYLE[fam]
        pts = []
        for b, seeds in pol[fam].items():
            arr = np.array(seeds)
            pts.append((arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 1].std(), len(seeds), b))
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]; ym = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
        nseed = pts[0][3]
        xs_s, ym_s = span(xs, ym)
        ax.plot(xs_s, ym_s, "o-", color=color, label=f"{lbl} (n={nseed})",
                markersize=5, linewidth=1.8, zorder=4)
        ax.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.15)

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
