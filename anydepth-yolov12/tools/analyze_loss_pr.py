"""
Loss vs per-image precision/recall scatter (Super & Base).

Usage:
    python tools/analyze_loss_pr.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_pr.csv \
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


def scatter(ax, x, y, title, color):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes); return
    pear, _ = pearsonr(x, y)
    spear, _ = spearmanr(x, y)
    ax.scatter(x, y, s=3, alpha=0.25, color=color, edgecolors="none")
    if x.std() > 1e-9:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, a * xs + b, color="black", lw=1.2, ls="--", label=f"y={a:.2f}x+{b:.2f}")
        ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{title}\nPearson={pear:+.3f}  Spearman={spear:+.3f}  (n={n})", fontsize=10)
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="analysis/bdd100k-AnyDepth/loss-pr")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"[*] n={len(df)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey="row")
    for ri, (tag, color) in enumerate([("super", "tab:blue"), ("base", "tab:red")]):
        loss = df[f"loss_{tag}"].to_numpy(dtype=float)
        for ci, metric in enumerate(["precision", "recall"]):
            x = df[f"{metric}_{tag}"].to_numpy(dtype=float)
            ax = axes[ri, ci]
            scatter(ax, x, loss, f"[{tag.title()}-net] loss vs {metric}", color)
            ax.set_xlabel(metric); ax.set_ylabel(f"loss_{tag}")
    fig.suptitle("Loss vs per-image precision / recall  (conf>=0.25, IoU>=0.5)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "loss_pr_corr.png", dpi=150)
    plt.close(fig)

    print("\n[*] correlation table (Pearson / Spearman):")
    print(f"{'metric':22s} {'Super_P':>9s} {'Super_S':>9s} {'Base_P':>9s} {'Base_S':>9s}")
    for metric in ["precision", "recall"]:
        row = []
        for tag in ["super", "base"]:
            x = df[f"{metric}_{tag}"].to_numpy(dtype=float)
            y = df[f"loss_{tag}"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            row += [pearsonr(x[m], y[m])[0], spearmanr(x[m], y[m])[0]]
        print(f"{metric:22s} {row[0]:+9.3f} {row[1]:+9.3f} {row[2]:+9.3f} {row[3]:+9.3f}")
    print(f"\n[*] figure -> {outdir / 'loss_pr_corr.png'}")


if __name__ == "__main__":
    main()
