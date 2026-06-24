"""Fig 10: Router design & grid resolution ablation (BDD100K, feat=both).
GAP-MLP vs TinyConv 2x2 vs TinyConv 8x8, mean±std over seeds.

Sources:
  - TinyConv 2x2 (policy_bn*)  + GAP-MLP (policy_gmbn*): video_curve_archabl.json
  - TinyConv 8x8 (policy_tinyconv_g8_s*): video_curve_router_grid_bdd.json

    python -m method_advantage_regress.ablation.make_router_grid_figure
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCHABL = "method_advantage_regress/outputs/bdd100k/eval/video_curve_archabl.json"
G8_JSON = "method_advantage_regress/outputs/bdd100k/eval/video_curve_router_grid_bdd.json"

STYLE = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
}


def collect(rows, fam_re, metric):
    """Group rows by budget/thres key, return list of (super%, AP, std)."""
    by = defaultdict(list)
    for r in rows:
        fam = r.get("family", "") or r.get("name", "")
        if re.match(fam_re, fam):
            key = r["budget"] if r.get("budget") is not None else r["thres"]
            by[key].append((r["super_rate"] * 100, r[metric] * 100))
    pts = []
    for key, seeds in by.items():
        a = np.array(seeds)
        pts.append((a[:, 0].mean(), a[:, 1].mean(), a[:, 1].std()))
    pts.sort()
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="map", choices=["map", "map50"])
    ap.add_argument("--out", default="method_advantage_regress/outputs/figures/fig_abl_router_grid_ap5095.pdf")
    args = ap.parse_args()
    metric = args.metric

    archabl = json.loads(Path(ARCHABL).read_text())
    g8     = json.loads(Path(G8_JSON).read_text())

    gb = archabl["gflops_base"]
    gs = archabl["gflops_super"]
    anchors = {r["name"]: (r["super_rate"] * 100, r[metric] * 100)
               for r in archabl["rows"] if r["name"] in ("always_base", "always_super")}

    # family regex -> (color, marker, label)
    CURVES = [
        (archabl["rows"], r"^policy_bn\d+$",            "tab:blue",  "s", r"TinyConv $2\times2$ (default)"),
        (g8["rows"],      r"^policy$|^policy_tinyconv",  "tab:green", "^", r"TinyConv $8\times8$"),
        (archabl["rows"], r"^policy_gmbn\d+$",           "tab:red",   "o", "GAP-MLP"),
    ]

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5.0, 3.7))

    for rows, fam_re, color, mk, lbl in CURVES:
        pts = collect(rows, fam_re, metric)
        if not pts:
            print(f"[!] no data for {lbl}")
            continue
        if "always_base" in anchors:
            pts.insert(0, (anchors["always_base"][0], anchors["always_base"][1], 0.0))
        if "always_super" in anchors:
            pts.append((anchors["always_super"][0], anchors["always_super"][1], 0.0))
        pts = sorted(pts)
        xs = np.array([p[0] for p in pts])
        ym = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        ax.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.15, zorder=2)
        ax.plot(xs, ym, marker=mk, color=color, label=lbl,
                markersize=4.5, linewidth=1.8, zorder=4)

    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="0.4", ls=":", lw=1, alpha=0.6)

    ax.set_xlabel("SUPER usage (%)")
    ax.set_ylabel(r"AP$_{50:95}$")
    ax.set_xlim(-3, 103)
    ax.legend(frameon=False, fontsize=13, loc="lower right")

    def s2g(s): return gb + (gs - gb) * s / 100.0
    def g2s(g): return (g - gb) / (gs - gb) * 100.0
    sec = ax.secondary_xaxis("top", functions=(s2g, g2s))
    sec.set_xlabel("GFLOPs / frame", fontsize=12)
    sec.tick_params(labelsize=11)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight", pad_inches=0.02)
    print(f"[*] -> {args.out}")


if __name__ == "__main__":
    main()
