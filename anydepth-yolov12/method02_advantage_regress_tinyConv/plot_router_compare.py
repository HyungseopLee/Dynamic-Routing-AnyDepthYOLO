"""Compare router designs under the HONEST pipeline (val-fixed thresholds).

Each curve point is a target SUPER budget whose threshold tau was fixed on the
val cache (get_thresholds.py); video eval only APPLIED tau (--val_taus_json).
We plot AP vs the ACTUAL video SUPER usage, averaged over policy seeds (std band).

Usage:
    python method02_advantage_regress_tinyConv/plot_router_compare.py \
        --curves "tiny-conv:method02_advantage_regress_tinyConv/outputs/kitti/eval/video_curve_tinyconv_backbone_valtau.json,GAP-MLP:method01_advantage_regress/outputs/kitti/eval/video_curve_mlp_backbone_valtau.json" \
        --metric map50 --out method02_advantage_regress_tinyConv/outputs/kitti/eval/fig_router_compare
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def aggregate(curve, metric):
    """budget -> (mean_video_super%, mean_metric, std_metric); plus endpoints, gflops."""
    d = json.loads(Path(curve).read_text())
    rows = d["rows"]
    byb = defaultdict(list)
    endpoints = {}
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            endpoints[r["name"]] = (r["super_rate"] * 100, r[metric])
        elif r.get("budget") is not None:
            byb[r["budget"]].append((r["super_rate"] * 100, r[metric]))
    pts = []
    for b in sorted(byb):
        a = np.array(byb[b])
        pts.append((a[:, 0].mean(), a[:, 1].mean(), a[:, 1].std()))
    pts.sort()
    return pts, endpoints, d["gflops_base"], d["gflops_super"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True, help="comma-sep tag:path list")
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Router design (val-fixed thresholds)")
    args = ap.parse_args()

    specs = [c.split(":", 1) for c in args.curves.split(",")]
    FMTS = ["o-", "s--", "^:", "D-."]
    COLORS = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

    fig, ax = plt.subplots(figsize=(9, 6))
    gb = gs = None
    endpoints = {}
    for i, (tag, path) in enumerate(specs):
        pts, ep, gb, gs = aggregate(path, args.metric)
        endpoints = endpoints or ep
        xs = [p[0] for p in pts]; ym = [p[1] for p in pts]; ys = [p[2] for p in pts]
        c = COLORS[i % len(COLORS)]
        ax.plot(xs, ym, FMTS[i % len(FMTS)], color=c, markersize=5, label=tag)
        ax.fill_between(xs, np.array(ym) - np.array(ys), np.array(ym) + np.array(ys),
                        color=c, alpha=0.15)

    for nm, (x, y) in endpoints.items():
        ax.scatter([x], [y], marker="*", s=220, zorder=5,
                   color="black" if nm == "always_super" else "darkgreen")
        ax.annotate(nm.replace("always_", ""), (x, y), textcoords="offset points", xytext=(6, 6))
    if "always_super" in endpoints:
        ax.axhline(endpoints["always_super"][1], color="black", ls=":", alpha=0.4)

    s2g = lambda sp: (sp / 100.0) * gs + (1 - sp / 100.0) * gb
    g2s = lambda g: (g - gb) / (gs - gb) * 100.0
    ax.secondary_xaxis("top", functions=(s2g, g2s)).set_xlabel("GFLOPs (per frame)")
    ax.set_xlabel("actual video SUPER usage (%)")
    ax.set_ylabel(args.metric.upper())
    ax.set_title(args.title, pad=28, fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    out = f"{args.out}_{ {'map50': 'ap50', 'map': 'ap5095'}[args.metric] }.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
