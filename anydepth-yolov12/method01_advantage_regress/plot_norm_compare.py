"""Compare BatchNorm vs LayerNorm policies on the same ablation axes.

Reads two eval_video outputs (one per norm choice) that share the same
policy_<feat>_s<seed>_t<tau> naming, aggregates seeds per (feat,tau) into a
mean curve, and overlays them: BatchNorm solid, LayerNorm dashed, colour=feat.

Usage:
    python method01_advantage_regress/plot_norm_compare.py --dataset kitti --metric map50
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

FEAT_COLOR = {"input": "tab:red", "pred": "tab:orange", "both": "tab:brown"}
FEAT_LBL = {"input": "backbone only", "pred": "neck only", "both": "both"}
NAME_RE = re.compile(r"policy_(input|pred|both)_s(\d+)_t([+-]\d+)")


def aggregate(curve, metric):
    """feat -> sorted list of (mean_super%, mean_metric, std_metric)."""
    rows = json.loads(Path(curve).read_text())["rows"]
    agg = defaultdict(lambda: defaultdict(list))
    endpoints = {}
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            endpoints[r["name"]] = (r["super_rate"] * 100, r[metric])
            continue
        m = NAME_RE.match(r["name"])
        if m:
            agg[m.group(1)][m.group(3)].append((r["super_rate"] * 100, r[metric]))
    out = {}
    for feat, taus in agg.items():
        pts = []
        for _tau, seeds in taus.items():
            a = np.array(seeds)
            pts.append((a[:, 0].mean(), a[:, 1].mean(), a[:, 1].std()))
        out[feat] = sorted(pts)
    return out, endpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--curves", default=None,
                    help="comma-sep tag:path list, e.g. 'BN:..seeds.json,none:..none.json'. "
                         "Default compares BN vs LN.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    ap.add_argument("--band", action="store_true", help="draw +/- std bands")
    ap.add_argument("--feats", default="input,pred,both",
                    help="comma-sep subset of feats to draw (e.g. 'input')")
    ap.add_argument("--title", default=None, help="plot title prefix (default 'Ablation')")
    args = ap.parse_args()
    keep = set(args.feats.split(","))
    ev = Path(__file__).resolve().parent / "outputs" / args.dataset / "eval"
    if args.curves is None:
        args.curves = (f"BN:{ev/'video_curve_featabl_seeds.json'},"
                       f"LN:{ev/'video_curve_featabl_ln.json'}")
    if args.out is None: args.out = str(ev / "norm_compare")

    specs = [c.split(":", 1) for c in args.curves.split(",")]
    FMTS = ["o-", "s--", "^:", "D-.", "v-", "P--", "X:", "*-."]
    stores = []
    endpoints = {}
    for i, (tag, path) in enumerate(specs):
        st, ep = aggregate(path, args.metric)
        stores.append((st, FMTS[i % len(FMTS)], tag))
        endpoints = endpoints or ep

    # one distinct colour per curve (tag); comparisons are usually same-feat so
    # colouring by feat would make every curve identical -> colour by tag instead.
    TAG_COLORS = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
                  "tab:brown", "tab:cyan", "tab:olive"]
    multi_feat = len(keep) > 1
    fig, ax = plt.subplots(figsize=(9, 6))
    for ti, (store, fmt, tag) in enumerate(stores):
        for feat, pts in store.items():
            if feat not in keep:
                continue
            c = FEAT_COLOR[feat] if multi_feat else TAG_COLORS[ti % len(TAG_COLORS)]
            lbl = f"{FEAT_LBL[feat]} [{tag}]" if multi_feat else tag
            xs = [p[0] for p in pts]; ym = [p[1] for p in pts]; ys = [p[2] for p in pts]
            ax.plot(xs, ym, fmt, color=c, markersize=4, label=lbl)
            if args.band:
                ax.fill_between(xs, np.array(ym) - np.array(ys), np.array(ym) + np.array(ys),
                                color=c, alpha=0.12)

    for nm, (x, y) in endpoints.items():
        ax.scatter([x], [y], marker="*", s=220, zorder=5,
                   color="black" if nm == "always_super" else "darkgreen")
        ax.annotate(nm.replace("always_", ""), (x, y), textcoords="offset points", xytext=(6, 6))
    if "always_super" in endpoints:
        ax.axhline(endpoints["always_super"][1], color="black", ls=":", alpha=0.4)

    d0 = json.loads(Path(specs[0][1]).read_text())
    gb, gs = d0["gflops_base"], d0["gflops_super"]
    s2g = lambda sp: (sp / 100.0) * gs + (1 - sp / 100.0) * gb
    g2s = lambda g: (g - gb) / (gs - gb) * 100.0
    ax.secondary_xaxis("top", functions=(s2g, g2s)).set_xlabel("GFLOPs (per frame)")
    ax.set_xlabel("SUPER usage (%)"); ax.set_ylabel(args.metric.upper())
    ax.set_title((args.title or "Ablation") + ": " + " vs ".join(t for _, _, t in stores),
                 pad=28, fontsize=10)
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    out = f"{args.out}_{ {'map50': 'ap50', 'map': 'ap5095'}[args.metric] }.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
