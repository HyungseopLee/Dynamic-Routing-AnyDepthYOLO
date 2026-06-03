"""Plot AP-vs-FLOPs Pareto curves from eval_video.py output.

Families: policy / conftop20 / confge10 / lum / edge / random, plus the
always-base / always-super endpoints. Each point can be annotated with its
threshold value.

Produces two figures:
  - <out>_all.png  : every family (both confidence variants drawn)
  - <out>_best.png : same, but only the single best confidence variant kept
                     (the weaker conf family is dropped). "best" = higher mean
                     AP across its operating points.

Usage:
    python method02_advantage_regress_tinyConv/plot_pareto.py \
        --curve runs/kitti/policy-eval/video_curve.json \
        --out runs/kitti/policy-eval/pareto
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = {
    "policy":        ("o-",  "tab:red",    "policy (A-hat)"),
    "policy_input":  ("o-",  "tab:red",    "policy: backbone feat"),
    "policy_pred":   ("o-",  "tab:orange", "policy: neck feat"),
    "policy_both":   ("o-",  "tab:brown",  "policy: both feat"),
    "conftop20":     ("s--", "tab:blue",   "conf: top-20 mean"),
    "confge10":      ("D--", "tab:cyan",   "conf: mean(>=0.1)"),
    "lum":           ("v-.", "tab:purple", "luminance"),
    "edge":          ("P-.", "tab:green",  "edge density"),
    "random":        ("^:",  "tab:gray",   "random"),
}


def collect(rows, metric):
    fams = {}
    endpoints = {}
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            endpoints[r["name"]] = (r["super_rate"] * 100, r[metric])
            continue
        k = r.get("family") or r.get("kind")
        if k in STYLES:
            fams.setdefault(k, []).append((r["super_rate"] * 100, r[metric], r.get("thres")))
    for k in fams:
        fams[k].sort()
    return fams, endpoints


def draw(fams, endpoints, metric, title, out, annotate, gb, gs, drop=()):
    fig, ax = plt.subplots(figsize=(9, 6))
    for fam_name, pts in fams.items():
        if not pts or fam_name in drop:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; ts = [p[2] for p in pts]
        fmt, color, lbl = STYLES[fam_name]
        ax.plot(xs, ys, fmt, color=color, label=lbl, markersize=5, linewidth=1.5)
        if annotate:
            for x, y, t in zip(xs, ys, ts):
                if t is None:
                    continue
                ax.annotate(f"{t:g}", (x, y), textcoords="offset points",
                            xytext=(0, 4), fontsize=6, color=color, ha="center")
    for nm, (x, y) in endpoints.items():
        ax.scatter([x], [y], marker="*", s=220, zorder=5,
                   color="black" if nm == "always_super" else "darkgreen")
        ax.annotate(nm.replace("always_", ""), (x, y),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    if "always_super" in endpoints:
        ax.axhline(endpoints["always_super"][1], color="black", ls=":", alpha=0.4,
                   label="super accuracy")
    ax.set_xlabel("SUPER usage (%)"); ax.set_ylabel(metric.upper())
    # top secondary axis: GFLOPs (linear map from super%)
    s2g = lambda sp: (sp / 100.0) * gs + (1 - sp / 100.0) * gb
    g2s = lambda g: (g - gb) / (gs - gb) * 100.0
    secax = ax.secondary_xaxis("top", functions=(s2g, g2s))
    secax.set_xlabel("GFLOPs (per frame)")
    ax.set_title(title, pad=28); ax.legend(); ax.grid(alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[*] saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti", help="output scope: outputs/<dataset>/eval/")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    ap.add_argument("--no-annotate", dest="annotate", action="store_false", default=True)
    args = ap.parse_args()
    _ev = Path(__file__).resolve().parent / "outputs" / args.dataset / "eval"
    if args.curve is None: args.curve = str(_ev / "video_curve.json")
    if args.out is None:   args.out = str(_ev / "pareto")

    data = json.loads(Path(args.curve).read_text())
    fams, endpoints = collect(data["rows"], args.metric)
    gb, gs = data["gflops_base"], data["gflops_super"]

    title = "AnyDepth depth routing: accuracy vs compute (KITTI-tracking)"
    # figure 1: all families
    draw(fams, endpoints, args.metric, title + " - all",
         f"{args.out}_all.png", args.annotate, gb, gs)

    # figure 2: keep only the best confidence variant
    conf_means = {k: (np.mean([p[1] for p in fams[k]]) if fams.get(k) else -1)
                  for k in ("conftop20", "confge10")}
    best_conf = max(conf_means, key=conf_means.get)
    drop = {c for c in ("conftop20", "confge10") if c != best_conf}
    print(f"[*] best confidence variant: {best_conf} "
          f"(mean {args.metric} {conf_means[best_conf]:.4f}); dropping {drop}")
    draw(fams, endpoints, args.metric, title + f" - best conf ({STYLES[best_conf][2]})",
         f"{args.out}_best.png", args.annotate, gb, gs, drop=drop)


if __name__ == "__main__":
    main()
