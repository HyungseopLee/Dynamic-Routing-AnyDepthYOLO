"""Main-result Pareto figure (paper style): AP vs SUPER usage (bottom x) with a twin
GFLOPs/frame top axis. The learned router is drawn as a mean+/-std band over seeds;
heuristic baselines (random, luminance, edge density, confidence) are optional dashed lines.

By default, produces TWO files from a single run:
  <out>_with_baselines.pdf  — random + lum + edge + conftop20 + ours
  <out>_router_only.pdf     — random + ours only

Use --dataset to auto-select curve path, router_fam, and out for each dataset.
Pass --show_baselines / --no_baselines to force a single output to --out.

Usage:
    # dataset shortcut (auto curve + router_fam + out)
    python -m analysis.make_main_figure --dataset kitti
    python -m analysis.make_main_figure --dataset bdd100k
    python -m analysis.make_main_figure --dataset waymo

    # all heuristic baselines + ours → single file
    python -m analysis.make_main_figure --dataset bdd100k \
        --show_baselines

    # random + ours only → single file
    python -m analysis.make_main_figure --dataset bdd100k \
        --no_baseline

    # shared legend strip (all baselines)
    python -m analysis.make_main_figure --dataset bdd100k \
        --legend_only --out results/figures/fig_legend.pdf

    # manual override (legacy)
    python -m analysis.make_main_figure \
        --curve results/step3_eval/bdd100k/eval/video_curve_archabl.json \
        --router_fam "router_bn(\\d+)" \
        --out results/figures/fig_main_bdd_ap5095.pdf
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


def _out_path(base_out: str, suffix: str) -> Path:
    p = Path(base_out)
    return p.parent / (p.stem + suffix + p.suffix)


def _draw_figure(ax, pol, anchors, bases, gb, gs, show_baselines: bool, no_legend: bool):
    """Draw one panel into ax. Returns nseed."""
    # random is always shown; full heuristics only when show_baselines=True
    extra_fams = ("lum", "edge", "conftop20") if show_baselines else ()
    for fam in ("random",) + extra_fams:
        if fam not in bases:
            continue
        pts = sorted(bases[fam])
        c, mk, ls, lbl = BASE_STYLE[fam]
        ax.plot([p[0] * 100 for p in pts], [p[1] for p in pts],
                marker=mk, color=c, ls=ls, label=lbl,
                markersize=4, linewidth=1.3, alpha=0.85)

    pts = []
    for key, seeds in pol.items():
        a = np.array(seeds)
        pts.append((a[:, 0].mean() * 100, a[:, 1].mean(), a[:, 1].std(), len(seeds)))
    pts.sort()
    nseed = pts[0][3] if pts else 0
    if "always_base" in anchors:
        pts.insert(0, (0.0, anchors["always_base"][1], 0.0, nseed))
    if "always_super" in anchors:
        pts.append((100.0, anchors["always_super"][1], 0.0, nseed))
    xs = np.array([p[0] for p in pts]); ym = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    ax.fill_between(xs, ym - ys, ym + ys, color="tab:red", alpha=0.18, zorder=3)
    ax.plot(xs, ym, "o-", color="tab:red", label="ours", markersize=4.5,
            linewidth=1.9, zorder=4)

    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="0.4", ls=":", lw=1, alpha=0.6)

    ax.set_xlabel("Super usage (\\%)" if matplotlib.rcParams["text.usetex"]
                  else "Super usage (%)")
    ax.set_ylabel(r"AP$_{50:95}$")
    ax.set_xlim(-3, 103)
    if not no_legend:
        ax.legend(frameon=False, fontsize=16, loc="lower right")

    def s2g(s):
        return gb + (s / 100.0) * (gs - gb)

    def g2s(g):
        return (g - gb) / (gs - gb) * 100.0

    sec = ax.secondary_xaxis("top", functions=(s2g, g2s))
    sec.set_xlabel("GFLOPs / frame")
    return nseed


# per-dataset defaults: (curve_path, router_fam_regex, out_path)
DATASET_DEFAULTS = {
    "kitti": (
        "results/step3_eval/kitti/eval/video_curve_main_both_g2.json",
        r"router_both_s(\d+)",
        "results/figures/fig_main_kitti_ap5095.pdf",
    ),
    "bdd100k": (
        "results/step3_eval/bdd100k/eval/video_curve_archabl.json",
        r"router_bn(\d+)",
        "results/figures/fig_main_bdd_ap5095.pdf",
    ),
    "waymo": (
        "results/step3_eval/waymo/eval_both/video_curve.json",
        r"router_seed(\d+)",
        "results/figures/fig_main_waymo_ap5095.pdf",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_DEFAULTS), default=None,
                    help="auto-select curve, router_fam, and out for a dataset")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--router_fam", default=None,
                    help="regex matching the router family; group(1) = seed id")
    ap.add_argument("--metric", default="map", choices=["map", "map50"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_legend", action="store_true")
    ap.add_argument("--legend_only", action="store_true",
                    help="render ONLY a shared horizontal legend strip to --out and exit")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--show_baselines", action="store_true",
                      help="force single output WITH heuristic baselines to --out")
    mode.add_argument("--no_baselines", action="store_true",
                      help="force single output WITHOUT heuristic baselines to --out")
    args = ap.parse_args()

    # resolve dataset defaults; explicit flags override
    if args.dataset:
        d_curve, d_fam, d_out = DATASET_DEFAULTS[args.dataset]
        if args.curve is None:
            args.curve = d_curve
        if args.router_fam is None:
            args.router_fam = d_fam
        if args.out is None:
            args.out = d_out
    else:
        if args.curve is None:
            ap.error("--curve is required when --dataset is not specified")
        if args.router_fam is None:
            args.router_fam = r"router_bn(\d+)"
        if args.out is None:
            args.out = "results/figures/fig_main_ap5095.pdf"

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
                         "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

    if args.legend_only:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color="tab:red", marker="o", ls="-", lw=1.9,
                          markersize=5, label="ours")]
        for fam in ("random", "lum", "edge", "conftop20"):
            c, mk, ls, lbl = BASE_STYLE[fam]
            handles.append(Line2D([0], [0], color=c, marker=mk, ls=ls, lw=1.3,
                                  markersize=5, label=lbl))
        figL = plt.figure(figsize=(9.0, 0.5))
        figL.legend(handles=handles, loc="center", ncol=len(handles),
                    frameon=False, fontsize=13, handlelength=2.2, columnspacing=1.6)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        figL.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
        print(f"[*] shared legend -> {args.out}")
        return

    curve_path = Path(args.curve)
    if not curve_path.exists():
        ap.error(f"curve file not found: {args.curve}\n"
                 f"  Run eval_video.py first to generate it.")
    data = json.loads(curve_path.read_text())
    rows = data["rows"]
    gb, gs = data["gflops_base"], data["gflops_super"]
    pol_re = re.compile(args.router_fam)
    AP = lambda r: r[args.metric] * 100.0

    bases = defaultdict(list)
    pol = defaultdict(list)
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.show_baselines or args.no_baselines:
        # forced single-output mode
        fig, ax = plt.subplots(figsize=(5.0, 3.7))
        nseed = _draw_figure(ax, pol, anchors, bases, gb, gs,
                             show_baselines=not args.no_baselines,
                             no_legend=args.no_legend)
        fig.tight_layout()
        fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"[*] router seeds={nseed} -> {args.out}")
    else:
        # default: produce both variants
        for show, suffix in ((True, "_with_baselines"), (False, "_router_only")):
            fig, ax = plt.subplots(figsize=(5.0, 3.7))
            nseed = _draw_figure(ax, pol, anchors, bases, gb, gs,
                                 show_baselines=show, no_legend=args.no_legend)
            fig.tight_layout()
            out = _out_path(args.out, suffix)
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            print(f"[*] router seeds={nseed} -> {out}")


if __name__ == "__main__":
    main()
