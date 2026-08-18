"""True advantage A distribution: overlaid histogram for KITTI / BDD100K / Waymo.

A = loss_base - loss_super  (per frame, from val cache)

Usage:
    python -m paper.make_advantage_dist
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RCPARAMS = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
}

DATASETS = [
    dict(
        label="KITTI",
        cache="results/step2_router/cache/kitti/cache_val_g2.pt",
        color="tab:red",
        bins=120,
    ),
    dict(
        label="BDD100K",
        cache="results/step2_router/cache/bdd100k/cache_val_both.pt",
        color="tab:blue",
        bins=200,
    ),
    dict(
        label="Waymo",
        cache="results/step2_router/cache/waymo/cache_val_both.pt",
        color="tab:green",
        bins=600,
    ),
]

OUT = "results/figures/fig_advantage_dist.pdf"


def load_A(cache_path):
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    return (c["loss_base"].view(-1) - c["loss_super"].view(-1)).numpy()


def main():
    plt.rcParams.update(RCPARAMS)

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    ax.axvline(0, color="0.5", lw=1.0, ls=":", zorder=1)

    for ds in DATASETS:
        if not Path(ds["cache"]).exists():
            print(f"  [skip] {ds['cache']}")
            continue
        A = load_A(ds["cache"])
        sigma = A.std()
        print(f"  [{ds['label']}] n={len(A)}  mean={A.mean():.4f}  σ={sigma:.4f}"
              f"  >0: {(A > 0).mean()*100:.1f}%")
        ax.hist(A, bins=ds["bins"], density=True, histtype="stepfilled",
                color=ds["color"], alpha=0.28,
                edgecolor=ds["color"], linewidth=1.8,
                label=rf"{ds['label']} ($\sigma$={sigma:.2f})",
                zorder=2)

    ax.set_xlabel(r"true advantage $A=L_\mathrm{base}-L_\mathrm{super}$", fontsize=16)
    ax.set_ylabel("density", fontsize=16)
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(0, 5.8)
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    ax.grid(alpha=0.25, ls="--", color="0.75")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"  [*] -> {OUT}")


if __name__ == "__main__":
    main()
