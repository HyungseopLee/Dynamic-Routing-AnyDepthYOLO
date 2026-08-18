"""
Which loss component is the best proxy for AP?

Per image, v8DetectionLoss decomposes into:
  - box : CIoU box-regression loss
  - dfl : distribution-focal regression loss
  - cls : BCE classification loss
  - reg : box + dfl              (the full localization/regression term)
  - total: box + cls + dfl

This script correlates EACH component against per-image detection quality
(ap5095 primary, plus precision / recall) for both the Super-net and Base-net,
and ranks the components by how strongly they track AP. The component with the
strongest (most negative) AP correlation is the best single-scalar proxy to
drive the policy advantage.

Usage:
    python tools/analyze_loss_pr.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_loss_pr_conf.csv \
        --outdir ./analysis/bdd100k-AnyDepth/loss-pr
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# loss components (label -> how to build the per-image series for a given net tag)
COMPONENTS = ["total", "box", "dfl", "cls", "reg"]
# the detection-quality metrics we correlate the loss against
METRICS = ["ap5095", "precision", "recall"]
NETS = [("super", "tab:blue"), ("base", "tab:red")]


def comp_series(df, comp, tag):
    """Per-image loss series for component `comp` on net `tag` (super/base)."""
    if comp == "total":
        return df[f"loss_{tag}"].to_numpy(dtype=float)
    if comp == "reg":
        return (df[f"box_{tag}"].to_numpy(dtype=float)
                + df[f"dfl_{tag}"].to_numpy(dtype=float))
    return df[f"{comp}_{tag}"].to_numpy(dtype=float)


def corr(x, y):
    """Pearson, Spearman, n over finite-paired entries."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan"), float("nan"), len(x)
    return pearsonr(x, y)[0], spearmanr(x, y)[0], len(x)


def scatter(ax, x, y, title, color):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes); return
    pear, spear, n = corr(x, y)
    ax.scatter(x, y, s=3, alpha=0.25, color=color, edgecolors="none")
    if x.std() > 1e-9:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, a * xs + b, color="black", lw=1.2, ls="--")
    ax.set_title(f"{title}\nPearson={pear:+.3f}  Spearman={spear:+.3f}  (n={n})", fontsize=9)
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="analysis/bdd100k-AnyDepth/loss-pr")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"[*] n={len(df)}  components={COMPONENTS}  metrics={METRICS}\n")

    # ---- correlation tables (one per metric): component x net, Pearson/Spearman ----
    for metric in METRICS:
        print(f"[*] loss-component vs {metric}  (Pearson / Spearman):")
        print(f"{'component':10s} {'Super_P':>9s} {'Super_S':>9s} {'Base_P':>9s} {'Base_S':>9s}")
        ranking = []
        for comp in COMPONENTS:
            cells = []
            for tag, _ in NETS:
                x = comp_series(df, comp, tag)
                y = df[f"{metric}_{tag}"].to_numpy(dtype=float)
                p, s, _ = corr(x, y)
                cells += [p, s]
            # rank by mean |Spearman| across both nets (rank-based = robust monotonic proxy)
            mean_abs_s = np.nanmean([abs(cells[1]), abs(cells[3])])
            ranking.append((comp, mean_abs_s))
            print(f"{comp:10s} {cells[0]:+9.3f} {cells[1]:+9.3f} {cells[2]:+9.3f} {cells[3]:+9.3f}")
        if metric == "ap5095":
            ranking.sort(key=lambda t: (-t[1] if np.isfinite(t[1]) else 0))
            print("  -> best AP proxy (by mean |Spearman|): "
                  + ", ".join(f"{c}({v:.3f})" for c, v in ranking))
        print()

    # ---- scatter grid: rows = components, cols = nets; y-axis = ap5095 ----
    metric = "ap5095"
    fig, axes = plt.subplots(len(COMPONENTS), len(NETS),
                             figsize=(5.0 * len(NETS), 3.2 * len(COMPONENTS)))
    for ri, comp in enumerate(COMPONENTS):
        for ci, (tag, color) in enumerate(NETS):
            x = comp_series(df, comp, tag)
            y = df[f"{metric}_{tag}"].to_numpy(dtype=float)
            ax = axes[ri, ci]
            scatter(ax, x, y, f"[{tag.title()}] {comp}-loss vs {metric}", color)
            ax.set_xlabel(f"{comp}_loss_{tag}"); ax.set_ylabel(metric)
    fig.suptitle("Per-image loss component vs mAP@[0.5:0.95]  "
                 "(reg = box + dfl; cls = BCE)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = outdir / "loss_component_vs_ap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[*] figure -> {out}")


if __name__ == "__main__":
    main()
