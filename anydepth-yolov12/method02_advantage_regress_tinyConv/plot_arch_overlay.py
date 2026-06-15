오"""Overlay tinyConv vs GAP-MLP routers on a SINGLE axis (so the arch gap is
visible). color = router input (backbone/back+neck/neck); linestyle = arch
(tinyConv solid, GAP-MLP dashed). Baselines + always-base/super endpoints
overlaid as in plot_arch_compare.py.

    python method02_advantage_regress_tinyConv/plot_arch_overlay.py \
        --curve outputs/bdd100k/eval/video_curve_archabl.json \
        --out outputs/bdd100k/eval/fig_bdd_video_archabl_overlay --metric map50
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

BASE_STYLE = {
    "random":    ("x", "tab:gray",   "random"),
    "lum":       ("s", "tab:green",  "luminance"),
    "edge":      ("D", "tab:orange", "edge density"),
    "conftop20": ("^", "tab:blue",   "confidence (top-20)"),
}
FEAT_COLOR = {"bb": "tab:red", "bn": "tab:purple", "pn": "tab:brown"}
FEAT_LABEL = {"bb": "backbone", "bn": "backbone+neck", "pn": "neck"}
ARCH_LS = {"tc": "-", "gm": "--"}
ARCH_LABEL = {"tc": "tinyConv", "gm": "GAP-MLP"}
POLICY_RE = re.compile(r"^policy_(gm)?(bb|bn|pn)(\d+)_b(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    ap.add_argument("--title", default="BDD100K MOT-val depth routing (200 clips): tinyConv vs GAP-MLP")
    args = ap.parse_args()
    rows = json.loads(Path(args.curve).read_text())["rows"]

    bases = defaultdict(list)
    pol = defaultdict(lambda: defaultdict(list))   # (arch,feat) -> budget -> [(super%,metric)]
    anchors, g_base, g_super = {}, None, None
    for r in rows:
        x = r["super_rate"] * 100.0
        if r["name"] == "always_base":  g_base = r["gflops"]
        if r["name"] == "always_super": g_super = r["gflops"]
        if r["name"] in ("always_base", "always_super"):
            anchors[r["name"]] = (x, r[args.metric]); continue
        m = POLICY_RE.match(r["name"])
        if m:
            arch = "gm" if m.group(1) else "tc"
            pol[(arch, m.group(2))][int(m.group(4))].append((x, r[args.metric])); continue
        fam = r.get("family") or r.get("kind")
        if fam in BASE_STYLE:
            bases[fam].append((x, r[args.metric]))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ab, asu = anchors.get("always_base"), anchors.get("always_super")

    def span(xs, ys):
        if ab: xs, ys = [ab[0]] + list(xs), [ab[1]] + list(ys)
        if asu: xs, ys = list(xs) + [asu[0]], list(ys) + [asu[1]]
        return xs, ys

    # baselines (light dashed-gray-ish)
    for fam, pts in bases.items():
        pts = sorted(pts); fmt, color, _ = BASE_STYLE[fam]
        xs, ys = span([p[0] for p in pts], [p[1] for p in pts])
        ax.plot(xs, ys, fmt + ":", color=color, markersize=4, linewidth=1.0, alpha=0.5, zorder=2)

    # learned routers: color=input, linestyle=arch
    for arch in ("tc", "gm"):
        for feat in ("bb", "bn", "pn"):
            budgets = pol.get((arch, feat))
            if not budgets:
                continue
            pts = sorted((np.mean([p[0] for p in v]), np.mean([p[1] for p in v]),
                          np.std([p[1] for p in v])) for v in budgets.values())
            xs = [p[0] for p in pts]; ym = np.array([p[1] for p in pts]); sd = np.array([p[2] for p in pts])
            xs_s, ym_s = span(xs, ym)
            ax.plot(xs_s, ym_s, ARCH_LS[arch], marker="o", color=FEAT_COLOR[feat],
                    markersize=3.5, linewidth=1.8, zorder=4)
            # +/-1 std over the 5 seeds (the whole point: is the arch gap inside seed noise?)
            ax.fill_between(xs, ym - sd, ym + sd, color=FEAT_COLOR[feat], alpha=0.10, zorder=1)

    for nm, (x, y) in anchors.items():
        ax.scatter([x], [y], marker="*", s=240, zorder=6,
                   color="black" if nm == "always_super" else "darkgreen")
        ax.annotate(nm.replace("always_", ""), (x, y), textcoords="offset points", xytext=(6, 6))

    ax.set_xlabel("SUPER usage (%)"); ax.set_ylabel(args.metric.upper())
    ax.grid(alpha=0.3)
    if g_base is not None and g_super is not None:
        sec = ax.secondary_xaxis("top", functions=(
            lambda p: g_base + p / 100 * (g_super - g_base),
            lambda g: (g - g_base) / (g_super - g_base) * 100))
        sec.set_xlabel("GFLOPs (per frame)")

    # two-part legend: color=input, linestyle=arch
    feat_handles = [Line2D([], [], color=FEAT_COLOR[f], marker="o", linestyle="-",
                           label=FEAT_LABEL[f]) for f in ("bb", "bn", "pn")]
    arch_handles = [Line2D([], [], color="black", linestyle=ARCH_LS[a], label=ARCH_LABEL[a])
                    for a in ("tc", "gm")]
    base_handles = [Line2D([], [], color=BASE_STYLE[f][1], marker=BASE_STYLE[f][0],
                           linestyle=":", alpha=0.6, label=BASE_STYLE[f][2])
                    for f in ("random", "conftop20", "lum", "edge")]
    leg1 = ax.legend(handles=feat_handles + arch_handles, fontsize=8, loc="lower right",
                     title="router (color=input, style=arch)")
    ax.add_artist(leg1)
    ax.legend(handles=base_handles, fontsize=7.5, loc="upper left", title="baselines")

    ax.set_title(args.title, pad=24)
    out = f"{args.out}_{ {'map50':'ap50','map':'ap5095'}[args.metric] }.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
