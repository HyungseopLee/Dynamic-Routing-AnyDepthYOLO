"""
Super-net vs Base-net per-image scatter — for loss and mAP side by side.

Outputs a single 1x2 figure:
  panel 1: loss_super (x) vs loss_base (y), y=x, linear fit, Pearson/Spearman;
           color = Δ_loss = base - super.
  panel 2: ap5095_super (x) vs ap5095_base (y), y=x, linear fit, Pearson/Spearman;
           color = Δ_ap = super - base (>0 means Super better).

Usage:
    python tools/analyze_super_vs_base.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_loss_pr_conf.csv \
        --outdir ./analysis/bdd100k-AnyDepth/super-vs-base
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def joint_scatter(ax, x, y, xlabel, ylabel, title_prefix,
                  super_wins_when="below", colorbar_label="Δ"):
    """Super vs Base scatter with y=x, linear fit, and Δ-colored points.

    super_wins_when: 'below' means below y=x means Super is better (loss case),
                     'above' means above y=x means Super is better (AP case).
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    # Δ direction: positive = Super wins
    if super_wins_when == "below":
        d = y - x   # loss: y>x means Base worse -> Super wins -> Δ>0
    else:
        d = x - y   # AP: x>y means Super > Base -> Δ>0

    pear, _ = pearsonr(x, y)
    spear, _ = spearmanr(x, y)
    n_super = int((d > 0).sum())
    n_base  = int((d < 0).sum())

    vmax = float(np.percentile(np.abs(d), 99)) or 1e-3
    sc = ax.scatter(x, y, c=d, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=4, alpha=0.6, edgecolors="none")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="y = x")
    a, c = np.polyfit(x, y, 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, a * xs + c, color="darkgreen", lw=1.4,
            label=f"fit: y={a:.2f}x{c:+.2f}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}  (Pearson={pear:+.3f}, Spearman={spear:+.3f})",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.04,
            f"Super better: {n_super} ({n_super/n*100:.1f}%)\n"
            f"Base  better: {n_base } ({n_base /n*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    return pear, spear, n, n_super, n_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="analysis/bdd100k-AnyDepth/super-vs-base")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"[*] n={len(df)}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ---- panel 1: loss ----
    s_loss = df["loss_super"].to_numpy()
    b_loss = df["loss_base"].to_numpy()
    pl, sl, nl, ns_l, nb_l = joint_scatter(
        axes[0], s_loss, b_loss,
        xlabel="loss_super", ylabel="loss_base",
        title_prefix="Loss — Super vs Base",
        super_wins_when="below",
        colorbar_label="Δ = base − super  (>0: Super wins)")
    print(f"    loss : Pearson={pl:+.3f} Spearman={sl:+.3f} | "
          f"Super better={ns_l} ({ns_l/nl*100:.1f}%), Base better={nb_l} ({nb_l/nl*100:.1f}%)")

    # ---- panel 2: AP ----
    s_ap = df["ap5095_super"].to_numpy()
    b_ap = df["ap5095_base"].to_numpy()
    pa, sa, na, ns_a, nb_a = joint_scatter(
        axes[1], s_ap, b_ap,
        xlabel="ap5095_super", ylabel="ap5095_base",
        title_prefix="mAP@[0.5:0.95] — Super vs Base",
        super_wins_when="above",
        colorbar_label="Δ = super − base  (>0: Super wins)")
    print(f"    ap   : Pearson={pa:+.3f} Spearman={sa:+.3f} | "
          f"Super better={ns_a} ({ns_a/na*100:.1f}%), Base better={nb_a} ({nb_a/na*100:.1f}%)")

    fig.suptitle(f"Super-net vs Base-net per image  (n={len(df)})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "super_vs_base.png", dpi=150)
    plt.close(fig)
    print(f"\n[*] figure -> {outdir / 'super_vs_base.png'}")


if __name__ == "__main__":
    main()
