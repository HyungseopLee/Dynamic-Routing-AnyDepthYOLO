"""Router grid-resolution ablation figure (paper style).
Our router with G=1x1 vs 2x2 vs 8x8 pooled grids, mean±std over seeds.

BDD100K sources (feat=both throughout):
  - G=1x1 (router_gmbn*) + G=2x2 (router_bn*): video_curve_archabl.json
  - G=8x8 (router_tinyconv_g8_*): video_curve_router_grid_bdd.json

KITTI sources (feat=both throughout):
  - G=1x1 (policy_gap*): video_curve_gap.json
  - G=2x2 (policy_bn*): video_curve_featcomp.json
  - G=8x8 (router_input_s*): video_curve_tinyconv_g8.json

Usage:
    python -m step3_eval.ablation.make_router_grid_figure
    python -m step3_eval.ablation.make_router_grid_figure --dataset kitti \
        --out results/figures/fig_abl_router_grid_kitti_ap5095.pdf
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = {
    "bdd100k": {
        "curves": [
            ("results/step3_eval/bdd100k/eval/video_curve_archabl.json",
             r"^router_gmbn\d+$", "tab:red",   "o", r"$G=1\times1$"),
            ("results/step3_eval/bdd100k/eval/video_curve_archabl.json",
             r"^router_bn\d+$",   "tab:blue",  "s", r"$G=2\times2$ (default)"),
            ("results/step3_eval/bdd100k/eval/video_curve_router_grid_bdd.json",
             r"^(router_tinyconv|policy)$", "tab:green", "^", r"$G=8\times8$"),
        ],
        "anchor_json": "results/step3_eval/bdd100k/eval/video_curve_archabl.json",
    },
    "kitti": {
        "curves": [
            ("results/step3_eval/kitti/eval/video_curve_grid_abl_g1_g2.json",
             r"^policy_gap_s\d+$",   "tab:red",   "o", r"$G=1\times1$"),
            ("results/step3_eval/kitti/eval/video_curve_grid_abl_g1_g2.json",
             r"^policy_bn\d+$",      "tab:blue",  "s", r"$G=2\times2$ (default)"),
            ("results/step3_eval/kitti/eval/video_curve_grid_abl_g8.json",
             r"^policy_g8_s\d+$",    "tab:green", "^", r"$G=8\times8$"),
        ],
        "anchor_json": "results/step3_eval/kitti/eval/video_curve_grid_abl_g1_g2.json",
    },
}

STYLE = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
}

INTERP_X = np.linspace(0, 100, 200)
MARKER_X  = np.linspace(5, 95, 9)


def collect_interp(rows, fam_re, metric, anchors):
    by = defaultdict(list)
    for r in rows:
        fam = r.get("family", "") or r.get("name", "")
        if re.match(fam_re, fam):
            key = r["budget"] if r.get("budget") is not None else r["thres"]
            by[key].append((r["super_rate"] * 100, r[metric] * 100))
    if not by:
        return None, None, None
    pts = []
    for key, seeds in by.items():
        a = np.array(seeds)
        pts.append((a[:, 0].mean(), a[:, 1].mean(), a[:, 1].std()))
    pts.sort()
    if "always_base" in anchors:
        pts.insert(0, (anchors["always_base"][0], anchors["always_base"][1], 0.0))
    if "always_super" in anchors:
        pts.append((anchors["always_super"][0], anchors["always_super"][1], 0.0))
    pts = sorted(pts)
    raw_x = np.array([p[0] for p in pts])
    raw_m = np.array([p[1] for p in pts])
    raw_s = np.array([p[2] for p in pts])
    ym = np.interp(INTERP_X, raw_x, raw_m)
    ys = np.interp(INTERP_X, raw_x, raw_s)
    return INTERP_X, ym, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bdd100k", choices=["bdd100k", "kitti"])
    ap.add_argument("--metric", default="map", choices=["map", "map50"])
    ap.add_argument("--out", default="results/figures/fig_abl_router_grid_ap5095.pdf")
    args = ap.parse_args()
    metric = args.metric
    cfg = CONFIGS[args.dataset]

    anchor_d = json.loads(Path(cfg["anchor_json"]).read_text())
    gb = anchor_d["gflops_base"]
    gs = anchor_d["gflops_super"]
    anchors = {r["name"]: (r["super_rate"] * 100, r[metric] * 100)
               for r in anchor_d["rows"] if r["name"] in ("always_base", "always_super")}

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5.0, 3.7))

    for json_path, fam_re, color, mk, lbl in cfg["curves"]:
        if not Path(json_path).exists():
            print(f"[!] missing {json_path}, skipping {lbl}")
            continue
        rows = json.loads(Path(json_path).read_text())["rows"]
        xs, ym, ys = collect_interp(rows, fam_re, metric, anchors)
        if xs is None:
            print(f"[!] no data for {lbl}")
            continue
        ax.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.15, zorder=2)
        ax.plot(xs, ym, color=color, label=lbl, linewidth=1.8, zorder=4)
        mk_ym = np.interp(MARKER_X, xs, ym)
        ax.plot(MARKER_X, mk_ym, marker=mk, color=color, linestyle="none",
                markersize=4.5, zorder=5)

    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="0.4", ls=":", lw=1, alpha=0.6)

    ax.set_xlabel("SUPER usage (%)")
    ax.set_ylabel(r"AP$_{50}$" if metric == "map50" else r"AP$_{50:95}$")
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
