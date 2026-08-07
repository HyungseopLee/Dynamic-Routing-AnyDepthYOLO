"""Feature-source ablation figure (paper style): AP vs SUPER usage for the three
router feature sources (backbone / neck / both), each drawn as a mean+/-std band over
seeds, with a twin GFLOPs/frame top axis.

    python -m method_advantage_regress.ablation.make_feat_ablation_figure \
        --curve method_advantage_regress/outputs/bdd100k/eval/video_curve_archabl.json \
        --out method_advantage_regress/outputs/figures/fig_feat_ablation_bdd_ap5095.pdf
    python -m method_advantage_regress.ablation.make_feat_ablation_figure \
        --curve method_advantage_regress/outputs/kitti/eval/video_curve_featcomp.json \
        --dataset kitti \
        --out method_advantage_regress/outputs/figures/fig_feat_ablation_kitti_ap5095.pdf
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

# BDD100K family prefixes
FEATS_BDD = [
    ("router_bb", "tab:red",   "o", "backbone"),
    ("router_pn", "tab:blue",  "s", "neck"),
    ("router_bn", "tab:green", "^", "both (default)"),
]

# KITTI family prefixes (same experiment, different naming convention)
FEATS_KITTI = [
    ("policy_bb", "tab:red",   "o", "backbone"),
    ("policy_pn", "tab:blue",  "s", "neck"),
    ("policy_bn", "tab:green", "^", "both (default)"),
]

INTERP_X = np.linspace(0, 100, 200)  # common x grid for smooth, comparable curves


def collect_interp(rows, prefix, metric, anchors):
    """Collect mean±std over seeds, then interpolate to INTERP_X."""
    fam_re = re.compile(re.escape(prefix) + r"(\d+)$")
    by = defaultdict(list)
    for r in rows:
        if fam_re.fullmatch(r.get("family", "")):
            key = r["budget"] if r.get("budget") is not None else r["thres"]
            by[key].append((r["super_rate"], r[metric] * 100.0))
    if not by:
        return None, None, None
    pts = []
    for key, seeds in by.items():
        a = np.array(seeds)
        pts.append((a[:, 0].mean() * 100, a[:, 1].mean(), a[:, 1].std()))
    pts.sort()
    if "always_base" in anchors:
        pts.insert(0, (0.0, anchors["always_base"][1], 0.0))
    if "always_super" in anchors:
        pts.append((100.0, anchors["always_super"][1], 0.0))
    raw_x = np.array([p[0] for p in pts])
    raw_m = np.array([p[1] for p in pts])
    raw_s = np.array([p[2] for p in pts])
    ym = np.interp(INTERP_X, raw_x, raw_m)
    ys = np.interp(INTERP_X, raw_x, raw_s)
    return INTERP_X, ym, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--dataset", default="bdd100k", choices=["bdd100k", "kitti"])
    ap.add_argument("--metric", default="map", choices=["map", "map50"])
    ap.add_argument("--out", default="method_advantage_regress/outputs/figures/fig_abl_feat_ap5095.pdf")
    args = ap.parse_args()

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]
    gb, gs = data["gflops_base"], data["gflops_super"]
    anchors = {r["name"]: (r["super_rate"] * 100, r[args.metric] * 100.0)
               for r in rows if r["name"] in ("always_base", "always_super")}

    FEATS = FEATS_KITTI if args.dataset == "kitti" else FEATS_BDD

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(5.0, 3.7))

    MARKER_X = np.linspace(5, 95, 9)  # marker positions on the interpolated curve

    for prefix, color, mk, lbl in FEATS:
        xs, ym, ys = collect_interp(rows, prefix, args.metric, anchors)
        if xs is None:
            continue
        ax.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.15, zorder=2)
        ax.plot(xs, ym, color=color, label=lbl, linewidth=1.8, zorder=4)
        # markers at fixed positions so all curves have the same marker layout
        mk_ym = np.interp(MARKER_X, xs, ym)
        ax.plot(MARKER_X, mk_ym, marker=mk, color=color, linestyle="none",
                markersize=4.5, zorder=5)

    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="0.4", ls=":", lw=1, alpha=0.6)

    ax.set_xlabel("SUPER usage (%)")
    ax.set_ylabel(r"AP$_{50:95}$")
    ax.set_xlim(-3, 103)
    ax.legend(frameon=False, fontsize=14, loc="lower right")

    def s2g(s):
        return gb + (s / 100.0) * (gs - gb)

    def g2s(g):
        return (g - gb) / (gs - gb) * 100.0
    sec = ax.secondary_xaxis("top", functions=(s2g, g2s))
    sec.set_xlabel("GFLOPs / frame", fontsize=12)
    sec.tick_params(labelsize=10)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight", pad_inches=0.02)
    print(f"[*] -> {args.out}")


if __name__ == "__main__":
    main()
