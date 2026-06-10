"""Compare tinyConv (conv-2x2) vs GAP-MLP routers on the BDD image-val AP curve.

Reads an image_curve json that contains BOTH router archs (emitted by
eval_image_routing_bdd.py): tinyConv families bb/bn/pn and GAP-MLP families
gmbb/gmbn/gmpn. Draws, per feature group (color), tinyConv as a SOLID line and
GAP-MLP as a DASHED line, so the conv-vs-GAP gap is read off directly. Baselines
and always-base/super endpoints are overlaid as in plot_video_curve_both.py.

    python method02_advantage_regress_tinyConv/plot_arch_compare.py \
        --curve outputs/bdd100k/eval/image_curve_arch.json --out outputs/bdd100k/eval/fig_arch --metric map50
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_STYLE = {
    "random":    ("x", "tab:gray",   "random"),
    "lum":       ("s", "tab:green",  "luminance"),
    "edge":      ("D", "tab:orange", "edge density"),
    "conftop20": ("^", "tab:blue",   "confidence (top-20)"),
}
FEAT_COLOR = {"bb": "tab:red", "bn": "tab:purple", "pn": "tab:brown"}
FEAT_LABEL = {"bb": "backbone", "bn": "backbone+neck", "pn": "neck"}
# (regex group1 = arch prefix '' or 'gm', group2 = feat, then seed, budget)
POLICY_RE = re.compile(r"^policy_(gm)?(bb|bn|pn)(\d+)_b(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ab, asu = anchors.get("always_base"), anchors.get("always_super")

    def span(xs, ys):
        if ab: xs, ys = [ab[0]] + list(xs), [ab[1]] + list(ys)
        if asu: xs, ys = list(xs) + [asu[0]], list(ys) + [asu[1]]
        return xs, ys

    PANEL = [("tc", "tinyConv (conv-2x2)"), ("gm", "GAP-MLP")]
    for ax, (arch, title) in zip(axes, PANEL):
        # baselines (same on both panels) in light gray-ish dashed
        for fam, pts in bases.items():
            pts = sorted(pts); fmt, color, lbl = BASE_STYLE[fam]
            xs, ys = span([p[0] for p in pts], [p[1] for p in pts])
            ax.plot(xs, ys, fmt + "--", color=color, label=lbl, markersize=4, linewidth=1.1, alpha=0.65)
        # this arch's 3 feature routers, solid, color = feat
        for feat in ("bb", "bn", "pn"):
            budgets = pol.get((arch, feat))
            if not budgets:
                continue
            pts = sorted((np.array(s)[:, 0].mean(), np.array(s)[:, 1].mean(), np.array(s)[:, 1].std())
                         for s in [v for v in [budgets[b] for b in budgets]])
            xs = [p[0] for p in pts]; ym = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
            xs_s, ym_s = span(xs, ym)
            ax.plot(xs_s, ym_s, "-o", color=FEAT_COLOR[feat], label=f"router: {FEAT_LABEL[feat]}",
                    markersize=4, linewidth=1.9, zorder=4)
            ax.fill_between(xs, ym - ys, ym + ys, color=FEAT_COLOR[feat], alpha=0.13)
        for nm, (x, y) in anchors.items():
            ax.scatter([x], [y], marker="*", s=220, zorder=5,
                       color="black" if nm == "always_super" else "darkgreen")
            ax.annotate(nm.replace("always_", ""), (x, y), textcoords="offset points", xytext=(6, 6))
        ax.set_xlabel("SUPER usage (%)"); ax.set_title(title, pad=24)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        if g_base is not None and g_super is not None:
            sec = ax.secondary_xaxis("top", functions=(
                lambda p: g_base + p / 100 * (g_super - g_base),
                lambda g: (g - g_base) / (g_super - g_base) * 100))
            sec.set_xlabel("GFLOPs (per frame)")
    axes[0].set_ylabel(args.metric.upper())
    fig.suptitle("BDD100K image-val depth routing: tinyConv vs GAP-MLP", y=1.0, fontsize=13)
    out = f"{args.out}_{ {'map50':'ap50','map':'ap5095'}[args.metric] }.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
