"""
Loss vs BDD100K attributes (weather / scene / timeofday) — focused analysis.

Complements analyze_loss_correlation.py with:
  1) boxplots per attribute (distribution, not just means) for Super, Base, Diff
  2) Kruskal-Wallis H-test + epsilon-squared effect size per attribute
  3) loss_diff (base - super) analysis: where does Base-net specifically struggle?
  4) 2D heatmaps for attribute pairs (weather x timeofday, etc.)

Usage:
    python tools/analyze_attrs_loss.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_loss_pr_conf.csv \
        --attr /media/data/bdd100k_yolo/val/attributes.json \
        --outdir ./analysis/bdd100k-AnyDepth/attributes-loss
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal


WEATHER = {0: "clear", 1: "rainy", 2: "snowy", 3: "overcast", 4: "foggy", 5: "partly cloudy", -1: "undefined"}
SCENE   = {0: "city street", 1: "highway", 2: "residential", 3: "parking lot", 4: "tunnel", 5: "gas stations", -1: "undefined"}
TIME    = {0: "daytime", 1: "night", 2: "dawn/dusk", -1: "undefined"}
MAPS = {"weather": WEATHER, "scene": SCENE, "timeofday": TIME}

# drop "undefined" from grouped plots/stats (still kept in df)
DROP_UNDEFINED = True
MIN_GROUP_N = 30  # skip groups smaller than this in stats / mark in plots


def epsilon_squared(H, k, n):
    """Effect size for Kruskal-Wallis. Range ~ [0, 1]."""
    if n - k <= 0:
        return float("nan")
    return float((H - k + 1) / (n - k))


def group_stats(df, attr, value_col):
    """Return per-group (label, n, mean, median, std) and K-W test result."""
    name_map = MAPS[attr]
    labels = [name_map[k] for k in sorted(name_map) if (not DROP_UNDEFINED or k != -1)]
    groups, rows = [], []
    for lab in labels:
        v = df.loc[df[attr + "_name"] == lab, value_col].to_numpy()
        if len(v) >= MIN_GROUP_N:
            groups.append(v)
            rows.append((lab, len(v), float(v.mean()), float(np.median(v)), float(v.std(ddof=1))))
    H, p = (float("nan"), float("nan"))
    eps2 = float("nan")
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        eps2 = epsilon_squared(H, len(groups), n_total)
    return pd.DataFrame(rows, columns=["label", "n", "mean", "median", "std"]), H, p, eps2


def grouped_boxplot(ax, df, attr, super_col, base_col, ylabel="detection loss", ylim=None):
    """Side-by-side Super (blue) vs Base (red) boxplots for one attribute."""
    name_map = MAPS[attr]
    labels, data_s, data_b, ns = [], [], [], []
    for k in sorted(name_map):
        if DROP_UNDEFINED and k == -1:
            continue
        lab = name_map[k]
        mask = df[attr + "_name"] == lab
        n_k = int(mask.sum())
        if n_k < MIN_GROUP_N:
            continue
        labels.append(lab)
        data_s.append(df.loc[mask, super_col].to_numpy())
        data_b.append(df.loc[mask, base_col].to_numpy())
        ns.append(n_k)

    k = len(labels)
    xs = np.arange(k)
    w = 0.36
    pos_s = xs - w / 2 - 0.02
    pos_b = xs + w / 2 + 0.02

    bp_s = ax.boxplot(data_s, positions=pos_s, widths=w, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black", lw=1.2))
    bp_b = ax.boxplot(data_b, positions=pos_b, widths=w, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch in bp_s["boxes"]:
        patch.set_facecolor("tab:blue"); patch.set_alpha(0.6); patch.set_edgecolor("black")
    for patch in bp_b["boxes"]:
        patch.set_facecolor("tab:red");  patch.set_alpha(0.6); patch.set_edgecolor("black")

    means_s = [float(np.mean(v)) for v in data_s]
    means_b = [float(np.mean(v)) for v in data_b]
    ax.scatter(pos_s, means_s, marker="D", s=26, color="white", edgecolor="black", zorder=5)
    ax.scatter(pos_b, means_b, marker="D", s=26, color="white", edgecolor="black", zorder=5)

    if ylim is not None:
        ax.set_ylim(*ylim)
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    # mean value labels just above each box
    for x, m in zip(pos_s, means_s):
        ax.text(x, m + span * 0.01, f"{m:.2f}", ha="center", va="bottom",
                fontsize=7, color="tab:blue", fontweight="bold")
    for x, m in zip(pos_b, means_b):
        ax.text(x, m + span * 0.01, f"{m:.2f}", ha="center", va="bottom",
                fontsize=7, color="tab:red", fontweight="bold")
    # n labels just under x-axis category
    for x, n_k in zip(xs, ns):
        ax.text(x, y_lo - span * 0.04, f"n={n_k}", ha="center", va="top",
                fontsize=8, color="dimgray")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_title(attr, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    # single legend handle per color (just for Super/Base, not duplicated)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="tab:blue", alpha=0.6, edgecolor="black", label="Super"),
                       Patch(facecolor="tab:red",  alpha=0.6, edgecolor="black", label="Base")],
              loc="upper right", fontsize=9, framealpha=0.9)


# (component_key, super_col, base_col, diff_col, pretty_label)
COMPONENTS = {
    "total": ("loss_super", "loss_base", "loss_diff", "total loss"),
    "box":   ("box_super",  "box_base",  "box_diff",  "box (regression) loss"),
    "cls":   ("cls_super",  "cls_base",  "cls_diff",  "cls (classification) loss"),
    "dfl":   ("dfl_super",  "dfl_base",  "dfl_diff",  "dfl loss"),
}


def run_for_component(df, n, comp_key, outdir):
    super_col, base_col, diff_col, pretty = COMPONENTS[comp_key]

    # If diff column missing (older CSV), derive on the fly.
    if diff_col not in df.columns:
        df = df.copy()
        df[diff_col] = df[base_col] - df[super_col]

    metrics = [(super_col, "tab:blue"), (base_col, "tab:red"), (diff_col, "tab:green")]

    print(f"\n[*] === component: {comp_key} ({pretty}) ===")
    print("[*] Kruskal-Wallis (per attribute, per metric):")
    for mcol, _ in metrics:
        for attr in MAPS:
            _, H, p, eps2 = group_stats(df, attr, mcol)
            print(f"  {mcol:14s} ~ {attr:10s}: H={H:8.2f}  p={p:.2e}  eps^2={eps2:.4f}")

    # ===== Combined figure: 2 rows x 3 attrs =====
    all_vals = np.concatenate([df[super_col].to_numpy(), df[base_col].to_numpy()])
    y_lo = float(np.percentile(all_vals, 1))
    y_hi = float(np.percentile(all_vals, 99))

    fig, axes = plt.subplots(2, len(MAPS), figsize=(20, 11),
                             gridspec_kw=dict(height_ratios=[2.2, 1.0]))
    for j, attr in enumerate(MAPS):
        grouped_boxplot(axes[0, j], df, attr, super_col, base_col,
                        ylabel=pretty, ylim=(y_lo, y_hi))
    for j in range(1, len(MAPS)):
        axes[0, j].sharey(axes[0, 0])

    for j, attr in enumerate(MAPS):
        ax = axes[1, j]
        tbl, H, p, eps2 = group_stats(df, attr, diff_col)
        # preserve MAP-defined order (matches grouped_boxplot's order)
        order_map = {MAPS[attr][k]: i for i, k in enumerate(sorted(MAPS[attr])) if k != -1}
        tbl = tbl.assign(_o=tbl["label"].map(order_map)).sort_values("_o").drop(columns="_o")
        x = np.arange(len(tbl))
        ax.bar(x, tbl["mean"], yerr=tbl["std"] / np.sqrt(tbl["n"]),
               capsize=4, color="tab:green", alpha=0.75, edgecolor="black")
        ax.axhline(0, color="black", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(tbl["label"], fontsize=10)
        ax.set_xlim(-0.6, len(tbl) - 0.4)
        ax.set_ylabel(f"{base_col} − {super_col} (± SE)")
        ax.set_title(f"Base-net penalty ({comp_key})  |  K-W H={H:.1f}, p={p:.1e}, eps²={eps2:.3f}",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for xi, (m, nn) in enumerate(zip(tbl["mean"], tbl["n"])):
            ax.text(xi, m, f"{m:+.3f}", ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=8, fontweight="bold")
        ax.margins(y=0.30)
    for j in range(1, len(MAPS)):
        axes[1, j].sharey(axes[1, 0])

    fig.suptitle(f"{pretty} by attribute (n={n}; top: Super vs Base distribution, "
                 f"bottom: Base-net penalty)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    suffix = "" if comp_key == "total" else f"_{comp_key}"
    fig.savefig(outdir / f"attr_loss_diff{suffix}.png", dpi=150)
    plt.close(fig)

    # ===== 4) Interaction plots: for each attr pair, lines = level of "hue" attr =====
    # x-axis = attr1 categories, one line per attr2 category, y = mean(loss_super|base)
    # Parallel lines => no interaction (effects add independently).
    # Crossing / fanning lines => interaction (one factor's effect depends on the other).
    pairs = list(combinations(MAPS.keys(), 2))  # (weather,scene), (weather,timeofday), (scene,timeofday)
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), 5.5))
    hue_cmap = plt.get_cmap("tab10")
    for ax, (a1, a2) in zip(axes, pairs):
        sub = df.copy()
        if DROP_UNDEFINED:
            sub = sub[(sub[a1] != -1) & (sub[a2] != -1)]

        x_levels = [MAPS[a1][k] for k in sorted(MAPS[a1])
                    if k != -1 and (sub[a1 + "_name"] == MAPS[a1][k]).sum() >= MIN_GROUP_N]
        hue_levels = [MAPS[a2][k] for k in sorted(MAPS[a2])
                      if k != -1 and (sub[a2 + "_name"] == MAPS[a2][k]).sum() >= MIN_GROUP_N]
        x_idx = np.arange(len(x_levels))

        for hi, hlab in enumerate(hue_levels):
            color = hue_cmap(hi % 10)
            mean_s, mean_b, ns = [], [], []
            for xlab in x_levels:
                cell = sub[(sub[a1 + "_name"] == xlab) & (sub[a2 + "_name"] == hlab)]
                if len(cell) < MIN_GROUP_N:
                    mean_s.append(np.nan); mean_b.append(np.nan); ns.append(len(cell))
                else:
                    mean_s.append(cell[super_col].mean())
                    mean_b.append(cell[base_col].mean())
                    ns.append(len(cell))
            ax.plot(x_idx, mean_s, "-o",  color=color, lw=1.8, ms=6, label=f"{hlab} (Super)")
            ax.plot(x_idx, mean_b, "--s", color=color, lw=1.5, ms=5, alpha=0.85,
                    label=f"{hlab} (Base)")
            # annotate N below the Super marker where defined
            for xi, (m, nv) in enumerate(zip(mean_s, ns)):
                if not np.isnan(m):
                    ax.text(xi, m, f"  n={nv}", fontsize=6, color=color,
                            ha="left", va="center")

        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_levels, rotation=20, ha="right", fontsize=9)
        ax.set_xlabel(a1)
        ax.set_ylabel(f"mean {pretty}")
        ax.set_title(f"{a1} × {a2}\n(line color = {a2}; solid=Super, dashed=Base)",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=1, framealpha=0.9, title=a2)
    fig.suptitle(f"Interaction plots ({pretty}) — parallel lines = independent effects; "
                 "fanning/crossing = interaction "
                 f"(cells with n<{MIN_GROUP_N} skipped)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / f"attr_interaction{suffix}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--attr", required=True)
    ap.add_argument("--outdir", default="runs/loss_analysis")
    ap.add_argument("--component", default="all",
                    choices=["all", "total", "box", "cls", "dfl"],
                    help="which loss component to analyze; 'all' runs every component")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    attrs = json.load(open(args.attr))
    for a in ("weather", "scene", "timeofday"):
        df[a] = df["stem"].map(lambda s, a=a: attrs.get(s, {}).get(a, -1))
        df[a + "_name"] = df[a].map(MAPS[a])

    n = len(df)
    print(f"[*] n={n}")

    comps = list(COMPONENTS.keys()) if args.component == "all" else [args.component]
    for c in comps:
        run_for_component(df, n, c, outdir)

    print(f"\n[*] figures written to {outdir}")


if __name__ == "__main__":
    main()
