"""
Loss vs post-NMS confidence correlation (Step 3).

For each model (Super, Base) and each conf definition
    mean_conf_25, mean_conf_50, mean_conf_75, top{10,20,30}_mean_conf
plots loss vs conf scatter with Pearson / Spearman.

Output: conf_loss_corr.png  (2 rows = Super/Base, 4 cols = conf definitions)

Usage:
    mkdir -p ./analysis/bdd100k-AnyDepth/conf-loss
    python tools/analyze_conf_loss.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_loss_pr_conf.csv \
        --outdir ./analysis/bdd100k-AnyDepth/conf-loss
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


CONF_DEFS = ["mean_conf_all", "mean_conf_01", "mean_conf_05",
             "mean_conf_10", "mean_conf_25", "mean_conf_50", "mean_conf_75",
             "top10_mean_conf", "top20_mean_conf", "top30_mean_conf"]


def scatter_panel(ax, x, y, title, color):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10); return
    pear, _ = pearsonr(x, y)
    spear, _ = spearmanr(x, y)
    ax.scatter(x, y, s=3, alpha=0.25, color=color, edgecolors="none")
    # linear fit
    if x.std() > 1e-9:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, a * xs + b, color="black", lw=1.2, ls="--",
                label=f"y={a:.2f}x+{b:.2f}")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{title}\nPearson={pear:+.3f}  Spearman={spear:+.3f}  (n={n})",
                 fontsize=10)
    ax.set_xlabel("confidence")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="analysis/bdd100k-AnyDepth/conf-loss")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"[*] n={len(df)}")

    fig, axes = plt.subplots(2, len(CONF_DEFS), figsize=(5.5 * len(CONF_DEFS), 10),
                             sharey="row")
    for ri, (tag, loss_col, color) in enumerate(
            [("Super", "loss_super", "tab:blue"), ("Base", "loss_base", "tab:red")]):
        for ci, conf in enumerate(CONF_DEFS):
            xcol = f"{conf}_{tag.lower()}"
            x = df[xcol].to_numpy(dtype=float)
            y = df[loss_col].to_numpy(dtype=float)
            scatter_panel(axes[ri, ci], x, y,
                          title=f"[{tag}-net] loss vs {conf}", color=color)
    fig.suptitle("Loss vs post-NMS confidence (per image)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "conf_loss_corr.png", dpi=150)
    plt.close(fig)

    # print correlation table
    print("\n[*] correlation table (Pearson / Spearman):")
    print(f"{'metric':24s} {'Super_P':>9s} {'Super_S':>9s} {'Base_P':>9s} {'Base_S':>9s}")
    for conf in CONF_DEFS:
        row = []
        for tag, loss_col in [("super", "loss_super"), ("base", "loss_base")]:
            x = df[f"{conf}_{tag}"].to_numpy(dtype=float)
            y = df[loss_col].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 2:
                row += [float("nan"), float("nan")]; continue
            row += [pearsonr(x[m], y[m])[0], spearmanr(x[m], y[m])[0]]
        print(f"{conf:24s} {row[0]:+9.3f} {row[1]:+9.3f} {row[2]:+9.3f} {row[3]:+9.3f}")

    print(f"\n[*] figure -> {outdir / 'conf_loss_corr.png'}")


if __name__ == "__main__":
    main()
