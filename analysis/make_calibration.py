"""Calibration figure: mean true advantage A per predicted-advantage rank bin.

Frames are sorted by ensemble-averaged Â and split into N_BINS equal groups.
Each point shows mean true A ± SEM for that bin, revealing whether higher Â
frames actually have higher A (monotone trend = well-calibrated router).

Usage:
    python -m analysis.make_calibration --dataset kitti
    python -m analysis.make_calibration --dataset bdd100k
    python -m analysis.make_calibration --dataset waymo
    python -m analysis.make_calibration  # all three
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from router.router_net import RouterNetwork, GapMlpNet

N_BINS = 10

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
        routers=[f"results/step3_eval/ablation/kitti/router_both_g2_s{s}.pt"
                 for s in range(5)],
        out="results/figures/fig_calibration_kitti.pdf",
    ),
    "bdd100k": dict(
        label="BDD100K",
        cache="results/step2_router/cache/bdd100k/cache_val_both.pt",
        routers=[f"results/step2_router/weights/bdd100k/router_both_{s}.pt"
                 for s in range(5)],
        out="results/figures/fig_calibration_bdd.pdf",
    ),
    "waymo": dict(
        label="Waymo",
        cache="results/step2_router/cache/waymo/cache_val_both.pt",
        routers=[
            "results/step2_router/weights/waymo/router_both_3.pt",
            "results/step2_router/weights/waymo/router_both_4.pt",
        ],
        out="results/figures/fig_calibration_waymo.pdf",
    ),
}


def _load_net(ckpt_path, inp, prd, dev):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    is_gap = not any(k.endswith("weight") and v.dim() == 4
                     for k, v in ck["state_dict"].items())
    cls = GapMlpNet if is_gap else RouterNetwork
    net = cls(group_dim=a.get("group_dim", 64),
              path_dim=a.get("path_dim", 8),
              hidden_dim=a.get("hidden", a.get("hidden_dim", 64)),
              feat=a.get("feat", "both"),
              norm=a.get("norm", "batch")).to(dev)
    pid2 = torch.zeros(2, dtype=torch.long, device=dev)
    with torch.no_grad():
        net.logit(inp[:2], prd[:2], pid2)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def _predict(net, inp, prd, pid):
    with torch.no_grad():
        return net.logit(inp, prd, pid).view(-1).cpu().numpy()


def draw_panel(ax, label, cache_path, router_paths, dev):
    c = torch.load(cache_path, map_location=dev, weights_only=False)
    inp = c["input_base"].to(dev).float()
    prd = c["pred_base"].to(dev).float()
    A   = (c["loss_base"].view(-1) - c["loss_super"].view(-1)).cpu().numpy()
    pid = torch.zeros(inp.shape[0], dtype=torch.long, device=dev)

    available = [p for p in router_paths if Path(p).exists()]
    if not available:
        print(f"  [!] no checkpoints found for {label}, skipping")
        return

    ahat = np.mean([_predict(_load_net(p, inp, prd, dev), inp, prd, pid)
                    for p in available], axis=0)
    r = float(np.corrcoef(A, ahat)[0, 1])
    print(f"  [{label}] n={len(A)}  seeds={len(available)}  r={r:.4f}")

    # sort by Â, split into N_BINS equal groups
    order = np.argsort(ahat)
    bins  = np.array_split(order, N_BINS)
    bx    = np.arange(1, N_BINS + 1)
    by    = np.array([A[b].mean() for b in bins])   # mean true A per bin
    be    = np.array([A[b].std() / np.sqrt(len(b)) for b in bins])

    ax.axhline(0, color="0.7", lw=0.9, ls=":", zorder=1)
    ax.errorbar(bx, by, yerr=be, fmt="o", color="tab:red",
                ecolor="tab:red", elinewidth=1.4, capsize=3,
                markersize=10, markeredgewidth=0, zorder=4)

    ax.set_xticks(bx)
    ax.set_xticklabels([str(i) for i in bx])
    ax.set_xlim(0.4, N_BINS + 0.6)
    ax.set_xlabel(r"rank bin  (frames sorted by $\hat{A}$)", fontsize=16)
    ax.set_ylabel(r"mean true advantage $A$", fontsize=16)
    ax.grid(alpha=0.25, ls="--", color="0.75")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_DEFAULTS), default=None)
    args = ap.parse_args()

    plt.rcParams.update(RCPARAMS)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    datasets = [args.dataset] if args.dataset else list(DATASET_DEFAULTS.keys())

    for ds in datasets:
        cfg = DATASET_DEFAULTS[ds]
        if not Path(cfg["cache"]).exists():
            print(f"  [skip] cache not found: {cfg['cache']}")
            continue
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        draw_panel(ax, cfg["label"], cfg["cache"], cfg["routers"], dev)
        Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(pad=0.5)
        fig.savefig(cfg["out"], bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  [*] -> {cfg['out']}")


if __name__ == "__main__":
    main()
