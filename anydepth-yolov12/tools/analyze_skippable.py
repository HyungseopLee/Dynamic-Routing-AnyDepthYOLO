"""
Δ = loss_base − loss_super distribution (decision-network label primer).

For each image:
    Δ <= ε   =>   "skippable"     (Base-net is close enough to Super-net)
    Δ >  ε   =>   "non-skippable" (collapsing depth costs too much)

Plots Δ histogram + sweep of skippable fraction vs ε.

Usage:
    python tools/analyze_skippable.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_loss_conf.csv \
        --eps 0.05 \
        --outdir ./analysis/bdd100k-AnyDepth/skippable
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS_SWEEP = [0.0, 0.025, 0.05, 0.10, 0.20]


def fig_delta(df, eps, outdir):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    delta = df["loss_diff"].to_numpy()
    n = len(delta)

    lo, hi = np.percentile(delta, [0.5, 99.5])
    counts, bins, _ = ax.hist(delta, bins=80, range=(lo, hi),
                              color="tab:purple", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", ls="--", lw=1.2)

    n_left  = int((delta < 0).sum())   # Base lower than Super (Super struggles more)
    n_right = int((delta > 0).sum())   # Base higher than Super (Base struggles more)
    n_zero  = int((delta == 0).sum())
    ymax = counts.max()
    ax.text(lo + (0 - lo) * 0.5, ymax * 0.92,
            f"Δ < 0\n(loss_super > loss_base)\nn = {n_left} ({n_left/n*100:.1f}%)",
            ha="center", va="top", fontsize=10, color="tab:blue", fontweight="bold")
    ax.text(0 + (hi - 0) * 0.5, ymax * 0.92,
            f"Δ > 0\n(loss_base > loss_super)\nn = {n_right} ({n_right/n*100:.1f}%)",
            ha="center", va="top", fontsize=10, color="tab:red", fontweight="bold")
    if n_zero:
        ax.text(0, ymax * 0.5, f" Δ=0: {n_zero}", fontsize=8, color="black")

    cmap = plt.get_cmap("tab10")
    for i, e in enumerate(EPS_SWEEP):
        share = float((delta <= e).mean())
        ax.axvline(e, color=cmap(i), ls=":", lw=1.2,
                   label=f"ε={e:.3f}  ({share*100:.1f}% skippable)")

    pcts = np.percentile(delta, [10, 50, 90])
    ax.set_xlabel("Δ = loss_base − loss_super")
    ax.set_ylabel("# images")
    ax.set_title(f"Δ distribution  (n={n}, mean={delta.mean():.3f}, "
                 f"P10={pcts[0]:.2f}, P50={pcts[1]:.2f}, P90={pcts[2]:.2f})",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "delta_distribution.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--eps", type=float, default=0.05,
                    help="skippable iff loss_diff <= eps")
    ap.add_argument("--outdir", default="analysis/bdd100k-AnyDepth/skippable")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    n_skip = int((df["loss_diff"] <= args.eps).sum())
    n = len(df)
    print(f"[*] n={n}, ε={args.eps}  ->  skippable={n_skip} ({n_skip/n*100:.1f}%)")
    fig_delta(df, args.eps, outdir)
    print(f"[*] figure written to {outdir / 'delta_distribution.png'}")


if __name__ == "__main__":
    main()
