"""Regenerate all paper figures in IEEE Access style.

Compact single-column panels, large internal fonts, NO in-axes title (the caption
carries the description), vector PDF output. Reads the same eval_video JSON curves
used during experiments and writes fig_*.pdf into paper/_extracted/.

Two figure kinds:
  - main    : one merged curve (policy 5-seed mean +/- std) over baselines
              (random / luminance / edge-density).
  - compare : several curves, each a policy mean +/- std band (one per JSON),
              for ablations (colour = curve).

Usage:
    python paper/make_figs.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------- style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 4.0,
    "axes.labelsize": 3.0,
    "axes.titlesize": 3.0,
    "xtick.labelsize": 2.0,
    "ytick.labelsize": 2.0,
    "legend.fontsize": 3.0,
    "lines.linewidth": 1.0,
    "lines.markersize": 2.0,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,   # editable/embeddable fonts for IEEE
    "ps.fonttype": 42,
})
FIGSIZE = (3.45, 2.65)          # single IEEE column panel
COLORS = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
MARKERS = ["o", "s", "^", "D", "v"]

ROOT = Path(__file__).resolve().parent.parent / "anydepth-yolov12"
M2 = ROOT / "method02_advantage_regress_tinyConv/outputs/kitti/eval"
M1 = ROOT / "method01_advantage_regress/outputs/kitti/eval"
OUT = Path(__file__).resolve().parent / "_extracted"

NAME_RE = re.compile(r"policy_(input|pred|both)_s(\d+)_t([+-]\d+)")
METRICS = {"map50": "ap50", "map": "ap5095"}
YLABEL = {"map50": "AP$_{50}$", "map": "AP$_{50:95}$"}

BASE_STYLE = {                 # baseline family -> (marker, color, label)
    "random": ("x", "0.5", "random"),
    "lum":    ("v", "tab:purple", "luminance"),
    "edge":   ("P", "tab:green", "edge density"),
}


def aggregate(path, metric, feat="input"):
    """JSON -> (sorted [(x_mean, y_mean, y_std)], endpoints) for the policy curve."""
    data = json.loads(Path(path).read_text())
    rows = data["rows"]
    taus = defaultdict(list)
    endpoints = {}
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            endpoints[r["name"]] = (r["super_rate"] * 100, r[metric])
            continue
        m = NAME_RE.match(r["name"])
        if m and m.group(1) == feat:
            taus[m.group(3)].append((r["super_rate"] * 100, r[metric]))
    pts = []
    for _t, seeds in taus.items():
        a = np.array(seeds)
        pts.append((a[:, 0].mean(), a[:, 1].mean(), a[:, 1].std()))
    return sorted(pts), endpoints, data["gflops_base"], data["gflops_super"]


def baselines(path, metric):
    rows = json.loads(Path(path).read_text())["rows"]
    fam = defaultdict(list)
    for r in rows:
        f = r.get("family") or r.get("kind")
        if f in BASE_STYLE:
            fam[f].append((r["super_rate"] * 100, r[metric]))
    return fam


def _decorate(ax, gb, gs, metric, endpoints):
    s2g = lambda sp: (sp / 100.0) * gs + (1 - sp / 100.0) * gb
    g2s = lambda g: (g - gb) / (gs - gb) * 100.0
    sec = ax.secondary_xaxis("top", functions=(s2g, g2s))
    sec.set_xlabel("GFLOPs / frame", labelpad=2, fontsize=8.0)
    ax.set_xlabel("SUPER usage (%)", labelpad=2, fontsize=8.0)
    ax.set_ylabel(YLABEL[metric], fontsize=8.0)
    ax.tick_params(axis="both", which="major", labelsize=7.6, length=2.5, width=0.7)
    sec.tick_params(axis="x", which="major", labelsize=7.2, length=2.5, width=0.7)
    for nm, (x, y) in endpoints.items():
        ax.scatter([x], [y], marker="*", s=140, zorder=6,
                   color="black" if nm == "always_super" else "darkgreen")
    if "always_super" in endpoints:
        ax.axhline(endpoints["always_super"][1], color="black", ls=":", alpha=0.4, lw=1)
    ax.grid(alpha=0.22, linestyle="--")


def plot_main(name, src):
    for metric in METRICS:
        pts, ep, gb, gs = aggregate(src, metric)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for fam, pl in baselines(src, metric).items():
            pl = sorted(pl)
            mk, c, lbl = BASE_STYLE[fam]
            ax.plot([p[0] for p in pl], [p[1] for p in pl], mk + "--", color=c,
                    label=lbl, lw=1.2, ms=4, alpha=0.85)
        xs = [p[0] for p in pts]; ym = [p[1] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ym, "o-", color="tab:red", label="ours", zorder=5)
        ax.fill_between(xs, np.array(ym) - np.array(ys), np.array(ym) + np.array(ys),
                        color="tab:red", alpha=0.18)
        _decorate(ax, gb, gs, metric, ep)
        ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.25,
              frameon=False, fontsize=7.0)
        _save(fig, f"{name}_{METRICS[metric]}")


def plot_compare(name, curves):
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        gb = gs = None; ep = {}
        for i, (lbl, src) in enumerate(curves):
            pts, ep, gb, gs = aggregate(src, metric)
            xs = [p[0] for p in pts]; ym = [p[1] for p in pts]; ys = [p[2] for p in pts]
            c = COLORS[i % len(COLORS)]
            ax.plot(xs, ym, marker=MARKERS[i % len(MARKERS)], ls="-", color=c, label=lbl)
            ax.fill_between(xs, np.array(ym) - np.array(ys), np.array(ym) + np.array(ys),
                            color=c, alpha=0.14)
        _decorate(ax, gb, gs, metric, ep)
        ncol = 2 if len(curves) >= 4 else 1
        ax.legend(loc="lower right", ncol=ncol, handlelength=1.3,
              columnspacing=0.75, borderaxespad=0.25, frameon=False,
              fontsize=7.0)
        _save(fig, f"{name}_{METRICS[metric]}")


def _save(fig, stem):
    fig.tight_layout(pad=0.4)
    out = OUT / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[*] {out.name}")


MAIN = {"fig_main": M2 / "video_curve_main_g2_merged.json"}

COMPARE = {
    "fig_abl_feat": [("backbone", M2 / "video_curve_tinyconv_g2.json"),
                     ("neck", M2 / "video_curve_pred_g2.json"),
                     ("both", M2 / "video_curve_both_g2.json")],
    "fig_abl_router_grid": [("GAP-MLP", M1 / "video_curve_selcorr.json"),
                             ("TinyConv 2$\\times$2", M2 / "video_curve_tinyconv_g2.json"),
                             ("TinyConv 4$\\times$4", M2 / "video_curve_tinyconv_backbone.json"),
                             ("TinyConv 8$\\times$8", M2 / "video_curve_tinyconv_g8.json")],
    "fig_abl_gridshape": [("2$\\times$2 (square)", M2 / "video_curve_tinyconv_g2.json"),
                          ("1$\\times$4 (rect)", M2 / "video_curve_g1x4.json")],
    "fig_abl_prevaction": [("with prev-action", M2 / "video_curve_tinyconv_g2.json"),
                           ("without", M2 / "video_curve_noprev_g2.json")],
    "fig_abl_norm": [("BatchNorm", M1 / "video_curve_selcorr.json"),
                     ("none", M1 / "video_curve_A3none.json"),
                     ("LayerNorm", M1 / "video_curve_A3layer.json")],
    "fig_abl_prevp": [("0.0", M1 / "video_curve_Gprevp00.json"),
                      ("0.25", M1 / "video_curve_Gprevp025.json"),
                      ("0.5", M1 / "video_curve_selcorr.json"),
                      ("0.75", M1 / "video_curve_Gprevp075.json"),
                      ("1.0", M1 / "video_curve_Gprevp10.json")],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in MAIN.items():
        plot_main(name, src)
    for name, curves in COMPARE.items():
        missing = [str(p) for _, p in curves if not Path(p).exists()]
        if missing:
            print(f"[!] skip {name}: missing {missing}")
            continue
        plot_compare(name, curves)


if __name__ == "__main__":
    main()
