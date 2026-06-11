"""Justify regressing the loss-advantage: predicted advantage vs realized AP gap.

Per-image AP is undefined, so we cannot scatter loss against AP frame-by-frame.
Instead we bin frames by the router's predicted advantage A-hat and, within each
bin, recompute the realized detection AP@[.50:.95] of the BASE and SUPER paths by
pooling that bin's match records (the only level at which AP is well-defined).
Plotting the realized gap (AP_super - AP_base) against the predicted advantage
shows that the scalar the router regresses actually ranks frames by how much the
deep path helps -- the premise of the whole method.

    conda run -n yolov12 python -m method02_advantage_regress_tinyConv.plot_advantage_apgap \
        --dump method02_advantage_regress_tinyConv/outputs/kitti/perframe_dump.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import eval_baseline_kitti as B


def pooled_map(frames, tag):
    matches, gt = [], {}
    for fr in frames:
        matches.extend(fr[f"match_{tag}"])
        for c, k in fr["gt_count"].items():
            gt[c] = gt.get(c, 0) + k
    _, m50, m5095 = B.dataset_map_multi_iou(matches, gt)
    return m50, m5095


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(Path(__file__).resolve().parent /
                    "outputs/kitti/perframe_dump.pkl"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent.parent /
                    "paper/_extracted/fig_advantage_apgap.pdf"))
    ap.add_argument("--bins", type=int, default=8)
    args = ap.parse_args()
    frames = pickle.load(open(args.dump, "rb"))["frames"]
    n = len(frames)

    # routing signal = predicted advantage available at BASE (decide whether to
    # upgrade). Bin frames into equal-count quantile bins of A-hat.
    ahat = np.array([fr["ahat_base"] for fr in frames])
    order = np.argsort(ahat)
    edges = np.linspace(0, n, args.bins + 1).astype(int)

    xs, gaps, base_ap, super_ap = [], [], [], []
    print(f"{'bin':>4}{'n':>6}{'A_hat':>9}{'AP_base':>9}{'AP_super':>9}{'gap':>8}")
    for bi in range(args.bins):
        idx = order[edges[bi]:edges[bi + 1]]
        sub = [frames[i] for i in idx]
        _, mb = pooled_map(sub, "base")
        _, ms = pooled_map(sub, "super")
        xa = float(ahat[idx].mean())
        xs.append(xa); base_ap.append(mb * 100); super_ap.append(ms * 100)
        gaps.append((ms - mb) * 100)
        print(f"{bi:>4}{len(idx):>6}{xa:>9.3f}{mb*100:>9.2f}{ms*100:>9.2f}"
              f"{(ms-mb)*100:>8.2f}")

    # Pearson correlation of predicted advantage vs realized AP gap (bin level)
    r = float(np.corrcoef(xs, gaps)[0, 1])
    print(f"\n[*] Pearson r (A_hat vs realized AP gap) = {r:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.axvline(0, color="0.6", lw=0.8, ls="--")
    ax.plot(xs, gaps, "o-", color="tab:red", lw=1.4, ms=4, zorder=5)
    ax.set_xlabel(r"predicted advantage $\hat{A}$ (bin mean)", fontsize=8.5)
    ax.set_ylabel(r"realized AP$_{50:95}$ gap" "\n" r"(SUPER $-$ BASE)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, ls="--")
    ax.text(0.04, 0.92, f"Pearson $r={r:.2f}$", transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))
    fig.tight_layout(pad=0.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print(f"[*] -> {args.out}")


if __name__ == "__main__":
    main()
