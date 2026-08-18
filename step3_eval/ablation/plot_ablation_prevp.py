"""Fig 11 (re-run): prev-action sampling ratio ablation with feat=both on KITTI.

Reads video_curve_prevp_both_g2.json produced by run_prevaction_abl.sh.
router family names: both_p00_s*, both_p05_s*, both_p10_s*
  -> prev_p = 0.0 / 0.5 / 1.0

    python -m step3_eval.ablation.plot_ablation_prevp \
        --curve results/step3_eval/bdd100k/eval/video_curve_prevp_bdd.json
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

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

# p-tag -> (color, label)
P_STYLE = {
    "p00": ("tab:red",   "$p=0.0$"),
    "p05": ("tab:green", "$p=0.5$ (default)"),
    "p10": ("tab:blue",  "$p=1.0$"),
}
# family prefix regex: both_p{tag}_s{seed}
FAM_RE = re.compile(r"^router_both_(p\d+)_s(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results/step3_eval/kitti/eval/video_curve_prevp_both_g2.json")
    ap.add_argument("--metric", default="map", choices=["map", "map50"])
    ap.add_argument("--out", default="results/figures/fig_abl_prevp_ap5095.pdf")
    args = ap.parse_args()

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]
    gb = data.get("gflops_base"); gs = data.get("gflops_super")
    AP = lambda r: r[args.metric] * 100.0

    anchors = {}
    by = defaultdict(lambda: defaultdict(list))  # ptag -> super% -> [AP]
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            anchors[r["name"]] = (r["super_rate"] * 100, AP(r)); continue
        m = FAM_RE.fullmatch(r.get("family", ""))
        if not m:
            continue
        ptag = m.group(1)
        key = r["budget"] if r.get("budget") is not None else r["thres"]
        by[ptag][key].append((r["super_rate"] * 100, AP(r)))

    fig, ax = plt.subplots(figsize=(5.0, 3.7))

    for ptag, color_label in P_STYLE.items():
        color, label = color_label
        pts = []
        for key, seeds in by[ptag].items():
            a = np.array(seeds)
            pts.append((a[:, 0].mean(), a[:, 1].mean(), a[:, 1].std()))
        if not pts:
            print(f"[!] no data for {ptag}")
            continue
        pts.sort()
        # prepend/append anchors
        if "always_base" in anchors:
            pts.insert(0, (anchors["always_base"][0], anchors["always_base"][1], 0.0))
        if "always_super" in anchors:
            pts.append((anchors["always_super"][0], anchors["always_super"][1], 0.0))
        xs = np.array([p[0] for p in pts])
        ym = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        ax.fill_between(xs, ym - ys, ym + ys, color=color, alpha=0.15, zorder=2)
        ax.plot(xs, ym, "o-", color=color, label=label, markersize=4.5, linewidth=1.8, zorder=4)

    if "always_super" in anchors:
        ax.axhline(anchors["always_super"][1], color="0.4", ls=":", lw=1, alpha=0.6)

    ax.set_xlabel("SUPER usage (%)")
    ax.set_ylabel(r"AP$_{50:95}$")
    ax.set_xlim(-3, 103)
    ax.legend(frameon=False, fontsize=14, loc="lower right")

    if gb is not None and gs is not None:
        def s2g(s): return gb + (gs - gb) * s / 100.0
        def g2s(g): return (g - gb) / (gs - gb) * 100.0
        sec = ax.secondary_xaxis("top", functions=(s2g, g2s))
        sec.set_xlabel("GFLOPs / frame", fontsize=12)
        sec.tick_params(labelsize=10)

    fig.tight_layout()
    out = args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[*] saved {out}")

    # print summary stats
    print(f"\n{'p-tag':<8} {'n_pts':>6} {'mean_AP_mean':>14} {'mean_AP_std':>12}")
    for ptag in ("p00", "p05", "p10"):
        pts = []
        for key, seeds in by[ptag].items():
            a = np.array(seeds)
            pts.append((a[:, 1].mean(), a[:, 1].std()))
        if pts:
            means = [p[0] for p in pts]; stds = [p[1] for p in pts]
            print(f"{ptag:<8} {len(pts):>6} {np.mean(means):>14.3f} {np.mean(stds):>12.4f}")


if __name__ == "__main__":
    main()
