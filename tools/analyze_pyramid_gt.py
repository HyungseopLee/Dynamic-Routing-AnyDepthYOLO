"""
Pyramid level (P3 / P4 / P5) confidence vs GT-size correlation.

Hypothesis to test:
    P3 conf  <->  small  GT count   (positive correlation)
    P4 conf  <->  medium GT count
    P5 conf  <->  large  GT count
    off-diagonal pairs should be weaker (or negative).

Conf metric: n_pred_ge25 per level (count of anchors with sigmoid(cls) >= 0.25).
GT side:     n_gt_small / n_gt_medium / n_gt_large (COCO area bins).

Both Super-net and Base-net paths are evaluated and shown side-by-side.

Figures:
  - pyramid_conf_vs_gt_corr.png   3x3 Spearman heatmap for Super and Base
  - scatter_diag.png              3x3 scatter (rows = level, cols = gt bin)
                                  with regression line and Spearman ρ / Pearson r

Usage:
    python tools/analyze_pyramid_gt.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_pyramid_conf.csv \
        --outdir ./analysis/bdd100k-AnyDepth/pyramid-gt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


LEVELS = ("P3", "P4", "P5")
GT_BINS = ("small", "medium", "large")
CONF_COL = "n_pred_ge10"   # per-level conf metric (anchors with max-conf >= 0.25)
# CONF_COL = "top10_mean_conf"   # per-level conf metric (anchors with max-conf >= 0.25)


def build_matrix(df, path_tag):
    """Return (3, 3) Spearman ρ matrix: rows = levels (P3/P4/P5), cols = gt bins."""
    M = np.full((3, 3), np.nan)
    for i, lvl in enumerate(LEVELS):
        x = df[f"{CONF_COL}_{lvl}_{path_tag}"].values
        for j, gb in enumerate(GT_BINS):
            y = df[f"n_gt_{gb}"].values
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            rho, _ = spearmanr(x, y, nan_policy="omit")
            M[i, j] = rho
    return M


def fig_heatmap(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, tag in zip(axes, ("super", "base")):
        M = build_matrix(df, tag)
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(3)); ax.set_xticklabels([f"n_gt_{g}" for g in GT_BINS])
        ax.set_yticks(range(3)); ax.set_yticklabels([f"{l} {CONF_COL}" for l in LEVELS])
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            color="white" if abs(v) > 0.5 else "black",
                            fontsize=10, fontweight="bold")
        ax.set_title(f"{tag.title()}-net", fontsize=11, fontweight="bold")
        # frame the diagonal cells
        for k in range(3):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=2))
    fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman ρ")
    fig.suptitle("Per-image: pyramid-level confidence vs GT size  (boxed = on-diagonal)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] figure -> {out_path}")


def fig_scatter(df, out_path, path_tag="super"):
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for i, lvl in enumerate(LEVELS):
        x_col = f"{CONF_COL}_{lvl}_{path_tag}"
        for j, gb in enumerate(GT_BINS):
            ax = axes[i, j]
            x = df[x_col].values
            y = df[f"n_gt_{gb}"].values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            ax.scatter(x, y, s=4, alpha=0.25, color="tab:blue", edgecolor="none")
            # regression line
            if len(x) > 2 and np.std(x) > 0:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, m * xs + b, color="tab:red", lw=1.2)
                rho, p_s = spearmanr(x, y)
                r_p, p_p = pearsonr(x, y)
                ax.text(0.02, 0.98,
                        f"ρ_S={rho:+.3f}\nr_P={r_p:+.3f}\nN={len(x)}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=9, family="monospace",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
            ax.set_xlabel(f"{lvl} {CONF_COL}", fontsize=9)
            ax.set_ylabel(f"n_gt_{gb}", fontsize=9)
            # highlight diagonal
            if i == j:
                for s in ax.spines.values():
                    s.set_edgecolor("black"); s.set_linewidth(2.0)
            ax.grid(alpha=0.3)
    fig.suptitle(f"Pyramid conf vs GT size  ({path_tag}-net)  "
                 "— diagonals are the hypothesized matches",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[*] figure -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[*] loaded {len(df)} rows from {args.csv}")
    # sanity: required cols
    needed = []
    for tag in ("super", "base"):
        needed += [f"{CONF_COL}_{lvl}_{tag}" for lvl in LEVELS]
    needed += [f"n_gt_{g}" for g in GT_BINS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing[:5]}...")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fig_heatmap(df, out / "pyramid_conf_vs_gt_corr.png")
    fig_scatter(df, out / "scatter_diag.png", path_tag="super")

    # console summary
    print("\n=== Spearman ρ matrix (rows = level conf, cols = gt bin) ===")
    for tag in ("super", "base"):
        print(f"\n[{tag}]")
        M = build_matrix(df, tag)
        print(f"           {'small':>8s} {'medium':>8s} {'large':>8s}")
        for i, lvl in enumerate(LEVELS):
            row = "   ".join(f"{M[i, j]:+.3f}" for j in range(3))
            print(f"  {lvl:>3s}    {row}")


if __name__ == "__main__":
    main()
