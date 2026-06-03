"""Plot feature-ablation Pareto with seed aggregation (mean +/- std).

Expects eval_video output where policy strategies are named
    policy_<feat>_s<seed>_t<tau>          e.g. policy_input_s3_t+020
(produced by passing --policies "input_s0=...,input_s1=...,...,both_s4=..." ).

For each feat (input/pred/both) it groups the 5 seeds by threshold tau, then
plots the per-tau mean curve (x = mean SUPER usage %, y = mean metric) with a
+/- std band, so overlapping bands => the feat difference is not significant.

Usage:
    python method01_advantage_regress/plot_ablation.py \
        --curve ./method01_advantage_regress/outputs/kitti/eval/video_curve_featabl_seeds.json \
        --out   method01_advantage_regress/outputs/kitti/eval/
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

FEAT_STYLE = {"input": ("tab:red", "backbone only"),
              "pred":  ("tab:orange", "neck only"),
              "both":  ("tab:brown", "both")}
NAME_RE = re.compile(r"policy_(input|pred|both)_s(\d+)_t([+-]\d+)")

# baseline families (drawn as single curves, no seeds) + their plot style.
# Confidence-routing (conftop20/confge10) is only drawn with --with_conf (main
# result); the ablation figure shows just the detection-independent baselines.
BASE_STYLE = {
    "lum":       ("v-.", "tab:purple", "luminance"),
    "edge":      ("P-.", "tab:green",  "edge density"),
    "random":    ("^:",  "tab:gray",   "random"),
}
CONF_STYLE = {
    "conftop20": ("s--", "tab:blue",   "conf: top-20 mean"),
    "confge10":  ("D--", "tab:cyan",   "conf: mean(>=0.1)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti", help="output scope: outputs/<dataset>/eval/")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    ap.add_argument("--with_conf", action="store_true",
                    help="also draw confidence-routing baselines (main result figure)")
    ap.add_argument("--feats", default="input,pred,both",
                    help="comma-sep subset of policy feats to draw (e.g. 'input' for main result)")
    ap.add_argument("--conf_curve", default=None,
                    help="separate eval json to read confidence-routing baselines from "
                         "(lets the heavy conf baseline be evaluated independently and merged "
                         "here). Defaults to --curve when omitted.")
    args = ap.parse_args()
    _ev = Path(__file__).resolve().parent / "outputs" / args.dataset / "eval"
    if args.curve is None: args.curve = str(_ev / "video_curve_featabl_seeds.json")
    if args.out is None:   args.out = str(_ev / "ablation_seeds")
    base_style = dict(BASE_STYLE, **CONF_STYLE) if args.with_conf else BASE_STYLE

    data = json.loads(Path(args.curve).read_text())
    rows = data["rows"]
    gb, gs = data["gflops_base"], data["gflops_super"]

    # feat -> tau -> list over seeds of (super_rate, metric, thres)
    agg = defaultdict(lambda: defaultdict(list))
    bases = defaultdict(list)   # baseline family -> list of (super_rate, metric, thres)
    endpoints = {}
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            endpoints[r["name"]] = (r["super_rate"] * 100, r[args.metric])
            continue
        m = NAME_RE.match(r["name"])
        if m:
            feat, _seed, tau = m.group(1), m.group(2), m.group(3)
            agg[feat][tau].append((r["super_rate"] * 100, r[args.metric], r.get("thres")))
            continue
        fam = r.get("family") or r.get("kind")
        if fam in base_style and fam not in CONF_STYLE:
            bases[fam].append((r["super_rate"] * 100, r[args.metric], r.get("thres")))

    # confidence-routing baselines: read from a separate eval json if given (modular
    # eval) else from the main curve. Merge degenerate variants (identical points,
    # e.g. at high --conf where top-20 == mean(>=0.1)) into one "confidence routing".
    if args.with_conf:
        conf_rows = json.loads(Path(args.conf_curve).read_text())["rows"] if args.conf_curve else rows
        conf_fam = defaultdict(list)
        for r in conf_rows:
            fam = r.get("family") or r.get("kind")
            if fam in CONF_STYLE:
                conf_fam[fam].append((r["super_rate"] * 100, r[args.metric], r.get("thres")))
        keys = sorted(conf_fam)
        if len(keys) == 2 and sorted(conf_fam[keys[0]]) == sorted(conf_fam[keys[1]]):
            bases["conftop20"] = conf_fam[keys[0]]            # identical -> single curve
            base_style["conftop20"] = (CONF_STYLE["conftop20"][0], CONF_STYLE["conftop20"][1],
                                       "confidence routing")
        else:
            for fam, pts in conf_fam.items():
                bases[fam] = pts; base_style[fam] = CONF_STYLE[fam]

    fig, ax = plt.subplots(figsize=(9, 6))
    # baselines first (drawn underneath the policy curves)
    for fam, pts in bases.items():
        pts = sorted(pts)
        fmt, color, lbl = base_style[fam]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, fmt, color=color, label=lbl, markersize=4, linewidth=1.2, alpha=0.85)

    keep_feats = set(args.feats.split(","))
    for feat, taus in agg.items():
        if feat not in keep_feats:
            continue
        color, lbl = FEAT_STYLE[feat]
        pts = []
        for tau, seeds in taus.items():
            arr = np.array([(s[0], s[1]) for s in seeds])   # [n_seed, 2]
            thr = np.mean([s[2] for s in seeds if s[2] is not None]) if seeds else None
            x = arr[:, 0].mean()
            y_mean = arr[:, 1].mean(); y_std = arr[:, 1].std()
            pts.append((x, y_mean, y_std, len(seeds), thr))
        pts.sort()
        xs = [p[0] for p in pts]; ym = [p[1] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ym, "o-", color=color, label=f"{lbl} (n={pts[0][3]})", markersize=4)
        ax.fill_between(xs, np.array(ym) - np.array(ys), np.array(ym) + np.array(ys),
                        color=color, alpha=0.18)
        # annotate the A-hat threshold (mean over seeds) at each operating point
        for x, y, _s, _n, thr in pts:
            if thr is not None:
                ax.annotate(f"{thr:g}", (x, y), textcoords="offset points", xytext=(0, 5),
                            fontsize=6, color=color, ha="center")

    for nm, (x, y) in endpoints.items():
        ax.scatter([x], [y], marker="*", s=220, zorder=5,
                   color="black" if nm == "always_super" else "darkgreen")
        ax.annotate(nm.replace("always_", ""), (x, y), textcoords="offset points", xytext=(6, 6))
    if "always_super" in endpoints:
        ax.axhline(endpoints["always_super"][1], color="black", ls=":", alpha=0.4)

    ax.set_xlabel("SUPER usage (%)"); ax.set_ylabel(args.metric.upper())
    s2g = lambda sp: (sp / 100.0) * gs + (1 - sp / 100.0) * gb
    g2s = lambda g: (g - gb) / (gs - gb) * 100.0
    sec = ax.secondary_xaxis("top", functions=(s2g, g2s)); sec.set_xlabel("GFLOPs (per frame)")
    title = ("Depth routing: policy vs baselines (KITTI-tracking)" if args.with_conf
             else "Feature ablation (mean +/- std over seeds)")
    ax.set_title(title, pad=28)
    ax.legend(); ax.grid(alpha=0.3)
    out = f"{args.out}_{ {'map50': 'ap50', 'map': 'ap5095'}[args.metric] }.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
