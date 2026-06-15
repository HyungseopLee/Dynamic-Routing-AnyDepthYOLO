"""tinyConv vs GAP-MLP, ONE panel per router input, y-zoomed to the router band
with +/-1 std (over 5 seeds) shaded. This is the figure that answers "is the
arch gap inside seed noise?" -- with only 2 curves per panel the bands don't
collide. tinyConv solid, GAP-MLP dashed; always-base/super drawn as h-lines.

    python method02_advantage_regress_tinyConv/plot_arch_byinput.py \
        --curve outputs/bdd100k/eval/video_curve_archabl.json \
        --out outputs/bdd100k/eval/fig_bdd_video_archabl_byinput --metric map50
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FEAT_LABEL = {"bb": "backbone", "bn": "backbone+neck", "pn": "neck"}
ARCH = {"tc": ("tinyConv", "-", "tab:red"), "gm": ("GAP-MLP", "--", "tab:blue")}
POLICY_RE = re.compile(r"^policy_(gm)?(bb|bn|pn)(\d+)_b(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    args = ap.parse_args()
    rows = json.loads(Path(args.curve).read_text())["rows"]

    pol = defaultdict(lambda: defaultdict(list))   # (arch,feat) -> budget -> [(super%,metric)]
    base = sup = g_base = g_super = None
    for r in rows:
        if r["name"] == "always_base":  base, g_base = r[args.metric], r["gflops"]
        if r["name"] == "always_super": sup, g_super = r[args.metric], r["gflops"]
        m = POLICY_RE.match(r["name"])
        if m:
            arch = "gm" if m.group(1) else "tc"
            pol[(arch, m.group(2))][int(m.group(4))].append((r["super_rate"] * 100.0, r[args.metric]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, feat in zip(axes, ("bb", "bn", "pn")):
        for arch in ("tc", "gm"):
            budgets = pol.get((arch, feat))
            if not budgets:
                continue
            pts = sorted((np.mean([p[0] for p in v]), np.mean([p[1] for p in v]),
                          np.std([p[1] for p in v])) for v in budgets.values())
            xs = np.array([p[0] for p in pts]); ym = np.array([p[1] for p in pts]); sd = np.array([p[2] for p in pts])
            name, ls, c = ARCH[arch]
            ax.plot(xs, ym, ls, marker="o", color=c, markersize=4, linewidth=1.9, label=name, zorder=4)
            ax.fill_between(xs, ym - sd, ym + sd, color=c, alpha=0.15, zorder=1,
                            label=f"{name} ±1 std")
        if base is not None:
            ax.axhline(base, color="darkgreen", ls=":", lw=1, alpha=0.7)
            ax.axhline(sup, color="black", ls=":", lw=1, alpha=0.7)
        ax.set_title(f"router input: {FEAT_LABEL[feat]}", pad=24)
        ax.set_xlabel("SUPER usage (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
        if g_base is not None:
            sec = ax.secondary_xaxis("top", functions=(
                lambda p: g_base + p / 100 * (g_super - g_base),
                lambda g: (g - g_base) / (g_super - g_base) * 100))
            sec.set_xlabel("GFLOPs (per frame)")
    axes[0].set_ylabel(args.metric.upper())
    # zoom y to the router band (ignore the always-base anchor far below)
    lo = min(np.mean([p[1] for p in v]) - np.std([p[1] for p in v])
             for f in ("bb", "bn", "pn") for a in ("tc", "gm") for v in pol[(a, f)].values())
    axes[0].set_ylim(lo - 0.0015, sup + 0.0012)
    fig.suptitle("BDD100K MOT-val (200 clips): tinyConv vs GAP-MLP per router input "
                 "(±1 std over 5 seeds; dotted = always-base / always-super)", y=1.0, fontsize=12)
    out = f"{args.out}_{ {'map50':'ap50','map':'ap5095'}[args.metric] }.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
