"""
When does Base-net beat Super-net on per-image AP@[0.5:0.95]?

Δ = ap5095_super - ap5095_base.
  Δ > 0 -> Super wins
  Δ < 0 -> Base wins
  Δ = 0 -> tie (kept as its own group)

Produces:
  - delta_distribution.png  : histogram of Δ + win counts
  - attr_distribution.png   : weather/scene/timeofday composition in each group
  - obj_conf_boxplots.png   : GT count / conf stats / n_pred / loss in each group

Usage:
    python tools/analyze_base_wins.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_loss_pr_conf.csv \
        --attr /media/data/bdd100k_yolo/val/attributes.json \
        --outdir ./analysis/bdd100k-AnyDepth/base-wins
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal


WEATHER = {0: "clear", 1: "rainy", 2: "snowy", 3: "overcast", 4: "foggy",
           5: "partly cloudy", -1: "undefined"}
SCENE   = {0: "city street", 1: "highway", 2: "residential", 3: "parking lot",
           4: "tunnel", 5: "gas stations", -1: "undefined"}
TIME    = {0: "daytime", 1: "night", 2: "dawn/dusk", -1: "undefined"}
ATTR_MAPS = {"weather": WEATHER, "scene": SCENE, "timeofday": TIME}

GROUP_ORDER = ["base_win", "tie", "super_win"]
GROUP_COLOR = {"base_win": "#d62728", "tie": "#7f7f7f", "super_win": "#1f77b4"}


def epsilon_squared(H, k, n):
    if n - k <= 0: return float("nan")
    return float((H - k + 1) / (n - k))


def assign_group(delta):
    if delta > 0: return "super_win"
    if delta < 0: return "base_win"
    return "tie"


def fig_delta_distribution(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    delta = df["delta_ap"].values
    bins = np.linspace(-1.0, 1.0, 81)
    ax.hist(delta, bins=bins, color="#888", edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color="black", lw=1.0, ls="--")

    n_total = len(df)
    n_base  = int((df["group"] == "base_win").sum())
    n_super = int((df["group"] == "super_win").sum())
    n_tie   = int((df["group"] == "tie").sum())
    mean_d  = float(np.nanmean(delta))

    ax.text(0.02, 0.98,
            f"N = {n_total}\n"
            f"base_win (Δ<0): {n_base}  ({n_base / n_total * 100:.1f}%)\n"
            f"super_win (Δ>0): {n_super}  ({n_super / n_total * 100:.1f}%)\n"
            f"tie (Δ=0): {n_tie}  ({n_tie / n_total * 100:.1f}%)\n"
            f"mean Δ = {mean_d:+.4f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    ax.set_xlabel("Δ = ap5095_super − ap5095_base  (per image)")
    ax.set_ylabel("# images")
    ax.set_title("Per-image AP gap: when does Base beat Super?",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[*] figure -> {out_path}")


def fig_attr_distribution(df, out_path):
    """For each attribute (weather/scene/timeofday), stacked bar of
    group composition WITHIN each attribute value: 'in clear weather what
    fraction of images is base_win vs super_win vs tie?'
    """
    attrs = list(ATTR_MAPS.keys())
    fig, axes = plt.subplots(1, len(attrs), figsize=(6 * len(attrs), 5),
                             squeeze=False)
    for ax, attr in zip(axes[0], attrs):
        name_map = ATTR_MAPS[attr]
        ct = pd.crosstab(df[attr], df["group"]).reindex(columns=GROUP_ORDER,
                                                        fill_value=0)
        # row-normalize to %
        row_total = ct.sum(axis=1).replace(0, np.nan)
        pct = ct.div(row_total, axis=0) * 100.0
        # use string labels & order by total count desc
        order = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
        pct = pct.reindex(order)
        labels = [f"{name_map.get(int(k), str(k))}\n(n={int(ct.loc[k].sum())})"
                  for k in order]

        bottom = np.zeros(len(order))
        for g in GROUP_ORDER:
            vals = pct[g].fillna(0).values
            ax.bar(range(len(order)), vals, bottom=bottom,
                   color=GROUP_COLOR[g], edgecolor="white",
                   linewidth=0.5, label=g)
            # annotate %
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 4:
                    ax.text(xi, b + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=8, color="white",
                            fontweight="bold")
            bottom += vals
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("% of images in category")
        ax.set_title(attr, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        # K-W test on Δ across this attribute's groups
        groups = [df.loc[df[attr] == k, "delta_ap"].dropna().values for k in order]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            n = sum(len(g) for g in groups); k = len(groups)
            eps2 = epsilon_squared(H, k, n)
            ax.text(0.98, 0.02,
                    f"K-W H={H:.1f}\np={p:.1e}\nε²={eps2:.3f}",
                    transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=8, family="monospace",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("Group composition by attribute "
                 "(within each category, how often does Base win?)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[*] figure -> {out_path}")


def fig_obj_conf_boxplots(df, out_path):
    """Compare per-image features across the 3 groups (base_win / tie / super_win)."""
    feats = [
        ("n_gt",                "# GT objects"),
        ("n_pred_all_super",    "# super predictions"),
        ("n_pred_all_base",     "# base predictions"),
        ("loss_super",          "loss (super)"),
        ("loss_base",           "loss (base)"),
        ("top10_mean_conf_super", "top-10 mean conf (super)"),
        ("top10_mean_conf_base",  "top-10 mean conf (base)"),
        ("mean_conf_25_super",  "mean_conf≥0.25 (super)"),
        ("mean_conf_25_base",   "mean_conf≥0.25 (base)"),
    ]
    n = len(feats); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows),
                             squeeze=False)
    for i, (col, label) in enumerate(feats):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        data = [df.loc[df["group"] == g, col].dropna().values for g in GROUP_ORDER]
        bp = ax.boxplot(data, labels=GROUP_ORDER, showfliers=False,
                        patch_artist=True, widths=0.6)
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLOR[g]); patch.set_alpha(0.6)
        # medians
        for xi, vals in enumerate(data, start=1):
            if len(vals):
                ax.text(xi, np.median(vals), f"{np.median(vals):.2f}",
                        ha="center", va="bottom", fontsize=7)
        # K-W
        nonempty = [g for g in data if len(g) >= 5]
        if len(nonempty) >= 2:
            H, p = kruskal(*nonempty)
            ntot = sum(len(g) for g in nonempty); k = len(nonempty)
            eps2 = epsilon_squared(H, k, ntot)
            ax.text(0.98, 0.97,
                    f"H={H:.1f} p={p:.1e} ε²={eps2:.3f}",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=7, family="monospace",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for j in range(n, rows * cols):
        r, c = divmod(j, cols); axes[r, c].axis("off")
    fig.suptitle("Per-image features by group "
                 "(red = Base wins, gray = tie, blue = Super wins)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[*] figure -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--attr", required=True,
                    help="bdd100k attributes.json (stem -> {weather,scene,timeofday})")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["ap5095_super", "ap5095_base"]).copy()
    df["delta_ap"] = df["ap5095_super"] - df["ap5095_base"]
    df["group"] = df["delta_ap"].apply(assign_group)

    attrs = json.load(open(args.attr))
    for a in ("weather", "scene", "timeofday"):
        df[a] = df["stem"].map(lambda s, a=a: attrs.get(s, {}).get(a, -1))

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fig_delta_distribution(df, out / "delta_distribution.png")
    fig_attr_distribution(df, out / "attr_distribution.png")
    fig_obj_conf_boxplots(df, out / "obj_conf_boxplots.png")

    # quick console summary
    print("\n=== group counts ===")
    print(df["group"].value_counts())
    print("\n=== group × timeofday (count) ===")
    print(pd.crosstab(df["timeofday"].map(TIME), df["group"]))


if __name__ == "__main__":
    main()
