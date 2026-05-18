"""
Per-image mAP@[0.5:0.95] vs BDD100K attributes (weather / scene / timeofday).

Mirrors analyze_loss_attrs.py but for AP instead of loss.
  - top row: Super vs Base AP boxplots per attribute category
  - bottom row: Base-net AP drop = ap_super - ap_base (>0 means Base struggles)
  - Kruskal-Wallis test + epsilon-squared per (metric, attribute)

Usage:
    python tools/analyze_attrs_AP.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_pr.csv \
        --attr /media/data/bdd100k_yolo/val/attributes.json \
        --outdir ./analysis/bdd100k-AnyDepth/ap-attributes
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


WEATHER = {0: "clear", 1: "rainy", 2: "snowy", 3: "overcast", 4: "foggy", 5: "partly cloudy", -1: "undefined"}
SCENE   = {0: "city street", 1: "highway", 2: "residential", 3: "parking lot", 4: "tunnel", 5: "gas stations", -1: "undefined"}
TIME    = {0: "daytime", 1: "night", 2: "dawn/dusk", -1: "undefined"}
MAPS = {"weather": WEATHER, "scene": SCENE, "timeofday": TIME}

DROP_UNDEFINED = True
MIN_GROUP_N = 30
AP_COL = "ap5095"


def epsilon_squared(H, k, n):
    if n - k <= 0: return float("nan")
    return float((H - k + 1) / (n - k))


def group_stats(df, attr, value_col):
    name_map = MAPS[attr]
    labels = [name_map[k] for k in sorted(name_map) if (not DROP_UNDEFINED or k != -1)]
    groups, rows = [], []
    for lab in labels:
        v = df.loc[df[attr + "_name"] == lab, value_col].to_numpy()
        v = v[np.isfinite(v)]
        if len(v) >= MIN_GROUP_N:
            groups.append(v)
            rows.append((lab, len(v), float(v.mean()), float(np.median(v)), float(v.std(ddof=1))))
    H, p, eps2 = float("nan"), float("nan"), float("nan")
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        eps2 = epsilon_squared(H, len(groups), n_total)
    return pd.DataFrame(rows, columns=["label", "n", "mean", "median", "std"]), H, p, eps2


def grouped_boxplot(ax, df, attr, ylim=None):
    """Side-by-side Super (blue) vs Base (red) AP boxplots."""
    name_map = MAPS[attr]
    labels, data_s, data_b, ns = [], [], [], []
    for k in sorted(name_map):
        if DROP_UNDEFINED and k == -1: continue
        lab = name_map[k]
        mask = df[attr + "_name"] == lab
        vs = df.loc[mask, f"{AP_COL}_super"].to_numpy(); vs = vs[np.isfinite(vs)]
        vb = df.loc[mask, f"{AP_COL}_base" ].to_numpy(); vb = vb[np.isfinite(vb)]
        n_k = int(mask.sum())
        if n_k < MIN_GROUP_N: continue
        labels.append(lab); data_s.append(vs); data_b.append(vb); ns.append(n_k)

    k = len(labels); xs = np.arange(k); w = 0.36
    pos_s = xs - w / 2 - 0.02; pos_b = xs + w / 2 + 0.02
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
    if ylim is not None: ax.set_ylim(*ylim)
    y_lo, y_hi = ax.get_ylim(); span = y_hi - y_lo
    for x, m in zip(pos_s, means_s):
        ax.text(x, m + span * 0.01, f"{m:.2f}", ha="center", va="bottom",
                fontsize=7, color="tab:blue", fontweight="bold")
    for x, m in zip(pos_b, means_b):
        ax.text(x, m + span * 0.01, f"{m:.2f}", ha="center", va="bottom",
                fontsize=7, color="tab:red", fontweight="bold")
    for x, n_k in zip(xs, ns):
        ax.text(x, y_lo - span * 0.04, f"n={n_k}", ha="center", va="top",
                fontsize=8, color="dimgray")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_title(attr, fontsize=12, fontweight="bold")
    ax.set_ylabel("mAP@[0.5:0.95]")
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="tab:blue", alpha=0.6, edgecolor="black", label="Super"),
                       Patch(facecolor="tab:red",  alpha=0.6, edgecolor="black", label="Base")],
              loc="lower right", fontsize=9, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--attr", required=True)
    ap.add_argument("--outdir", default="analysis/bdd100k-AnyDepth/ap-attributes")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    attrs = json.load(open(args.attr))
    for a in ("weather", "scene", "timeofday"):
        df[a] = df["stem"].map(lambda s, a=a: attrs.get(s, {}).get(a, -1))
        df[a + "_name"] = df[a].map(MAPS[a])
    # ap_diff: Super-net advantage = ap_super - ap_base (>0 means Base loses ground)
    df["ap_diff"] = df[f"{AP_COL}_super"] - df[f"{AP_COL}_base"]

    n = len(df)
    print(f"[*] n={n}")

    metrics = [f"{AP_COL}_super", f"{AP_COL}_base", "ap_diff"]
    print("\n[*] Kruskal-Wallis (per attribute, per metric):")
    for mcol in metrics:
        for attr in MAPS:
            _, H, p, eps2 = group_stats(df, attr, mcol)
            print(f"  {mcol:13s} ~ {attr:10s}: H={H:8.2f}  p={p:.2e}  eps^2={eps2:.4f}")

    # ===== Combined figure: 2 rows x 3 attrs =====
    fig, axes = plt.subplots(2, len(MAPS), figsize=(20, 11),
                             gridspec_kw=dict(height_ratios=[2.2, 1.0]))
    for j, attr in enumerate(MAPS):
        grouped_boxplot(axes[0, j], df, attr, ylim=(0, 1))
    for j in range(1, len(MAPS)):
        axes[0, j].sharey(axes[0, 0])

    for j, attr in enumerate(MAPS):
        ax = axes[1, j]
        tbl, H, p, eps2 = group_stats(df, attr, "ap_diff")
        order_map = {MAPS[attr][k]: i for i, k in enumerate(sorted(MAPS[attr])) if k != -1}
        tbl = tbl.assign(_o=tbl["label"].map(order_map)).sort_values("_o").drop(columns="_o")
        x = np.arange(len(tbl))
        ax.bar(x, tbl["mean"], yerr=tbl["std"] / np.sqrt(tbl["n"]),
               capsize=4, color="tab:green", alpha=0.75, edgecolor="black")
        ax.axhline(0, color="black", ls="--", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(tbl["label"], fontsize=10)
        ax.set_xlim(-0.6, len(tbl) - 0.4)
        ax.set_ylabel("ap_super − ap_base (± SE)")
        ax.set_title(f"Base-net AP drop  |  K-W H={H:.1f}, p={p:.1e}, eps²={eps2:.3f}",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for xi, m in zip(x, tbl["mean"]):
            ax.text(xi, m, f"{m:+.3f}", ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=8, fontweight="bold")
        ax.margins(y=0.30)
    for j in range(1, len(MAPS)):
        axes[1, j].sharey(axes[1, 0])

    fig.suptitle(f"Per-image mAP@[0.5:0.95] by attribute (n={n}; top: Super vs Base, "
                 f"bottom: Base-net AP drop = ap_super − ap_base)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "attr_AP.png", dpi=150)
    plt.close(fig)
    print(f"\n[*] figure written to {outdir / 'attr_AP.png'}")


if __name__ == "__main__":
    main()
