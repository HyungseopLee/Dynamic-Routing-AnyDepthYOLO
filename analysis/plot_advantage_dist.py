"""True advantage A distribution histogram.

A = loss_base - loss_super  (per frame, from val cache)

Usage:
    python -m analysis.plot_advantage_dist
    python -m analysis.plot_advantage_dist --dataset kitti
    python -m analysis.plot_advantage_dist --dataset bdd100k
    python -m analysis.plot_advantage_dist --dataset waymo
"""
import argparse
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
    "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
}

DATASET_DEFAULTS = {
    "kitti": dict(
        label="KITTI",
        cache="results/step2_router/cache/kitti/cache_val_g2.pt",
        out="results/figures/fig_advantage_dist_kitti.pdf",
    ),
    "bdd100k": dict(
        label="BDD100K",
        cache="results/step2_router/cache/bdd100k/cache_val_both.pt",
        out="results/figures/fig_advantage_dist_bdd.pdf",
    ),
    "waymo": dict(
        label="Waymo",
        cache="results/step2_router/cache/waymo/cache_val_both.pt",
        out="results/figures/fig_advantage_dist_waymo.pdf",
    ),
}


def draw_panel(ax, label, cache_path):
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    A = (c["loss_base"].view(-1) - c["loss_super"].view(-1)).numpy()
    print(f"  [{label}] n={len(A)}  mean={A.mean():.4f}  std={A.std():.4f}"
          f"  >0: {(A > 0).mean()*100:.1f}%")

    q1, q99 = np.percentile(A, 1), np.percentile(A, 99)
    clip = A[(A >= q1) & (A <= q99)]

    ax.axvline(0, color="0.5", lw=1.0, ls=":", zorder=1)
    ax.hist(clip, bins=80, color="tab:blue", alpha=0.75, edgecolor="none", zorder=2)
    ax.set_xlabel(r"true advantage  $A = \ell_\mathrm{base} - \ell_\mathrm{super}$", fontsize=18)
    ax.set_ylabel("frame count", fontsize=18)
    ax.set_title(label, fontsize=18)
    ax.grid(alpha=0.25, ls="--", color="0.75")

    pct = (A > 0).mean() * 100
    ax.text(0.97, 0.96, f"{pct:.1f}% frames: super wins",
            transform=ax.transAxes, va="top", ha="right", fontsize=15,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_DEFAULTS), default=None)
    args = ap.parse_args()

    plt.rcParams.update(RCPARAMS)

    datasets = [args.dataset] if args.dataset else list(DATASET_DEFAULTS.keys())

    for ds in datasets:
        cfg = DATASET_DEFAULTS[ds]
        if not Path(cfg["cache"]).exists():
            print(f"  [skip] cache not found: {cfg['cache']}")
            continue
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        draw_panel(ax, cfg["label"], cfg["cache"])
        Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(pad=0.5)
        fig.savefig(cfg["out"], bbox_inches="tight", pad_inches=0.02)
        out_png = cfg["out"].replace(".pdf", ".png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  [*] -> {cfg['out']}")


if __name__ == "__main__":
    main()
