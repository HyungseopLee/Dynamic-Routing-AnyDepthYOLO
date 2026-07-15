"""Calibration figure: predicted advantage (Â) vs. true advantage (A).

Scatter-plots Â against A = L_base - L_super for each dataset, with a y=x
reference line and Pearson r annotation. Produces one PDF per dataset.

Usage:
    python -m method_advantage_regress.analysis.make_calibration --dataset kitti
    python -m method_advantage_regress.analysis.make_calibration --dataset bdd100k
    python -m method_advantage_regress.analysis.make_calibration --dataset waymo
    python -m method_advantage_regress.analysis.make_calibration  # all three
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from method_advantage_regress.router.router_net import RouterNetwork, GapMlpNet

RCPARAMS = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
}

DATASET_DEFAULTS = {
    "kitti": dict(
        label="KITTI",
        cache="method_advantage_regress/outputs/kitti/cache_val_g2.pt",
        routers=[
            f"method_advantage_regress/outputs/kitti/ablation/router_both_g2_s{s}.pt"
            for s in range(5)
        ],
        out="method_advantage_regress/outputs/figures/fig_calibration_kitti.pdf",
    ),
    "bdd100k": dict(
        label="BDD100K",
        cache="method_advantage_regress/outputs/bdd100k/cache_val_both.pt",
        routers=[
            f"method_advantage_regress/outputs/bdd100k/router_both_{s}.pt"
            for s in range(5)
        ],
        out="method_advantage_regress/outputs/figures/fig_calibration_bdd.pdf",
    ),
    "waymo": dict(
        label="Waymo",
        cache="method_advantage_regress/outputs/waymo/cache_val_both.pt",
        routers=[
            "method_advantage_regress/outputs/waymo/router_both_3.pt",
            "method_advantage_regress/outputs/waymo/router_both_4.pt",
        ],
        out="method_advantage_regress/outputs/figures/fig_calibration_waymo.pdf",
    ),
}


def _load_net(ckpt_path: str, dev: str):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    is_gap = not any(k.endswith("weight") and v.dim() == 4
                     for k, v in ck["state_dict"].items())
    cls = GapMlpNet if is_gap else RouterNetwork
    net = cls(group_dim=a.get("group_dim", 64),
              path_dim=a.get("path_dim", 8),
              hidden_dim=a.get("hidden_dim", 64),
              feat=a.get("feat", "both"),
              norm=a.get("norm", "batch")).to(dev)
    # materialize lazy layers
    c_dummy = torch.zeros(2, 1, 4, 4, device=dev)
    pid_dummy = torch.zeros(2, dtype=torch.long, device=dev)
    with torch.no_grad():
        net.logit(c_dummy, c_dummy, pid_dummy)
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
    A = (c["loss_base"].view(-1) - c["loss_super"].view(-1)).cpu().numpy()
    pid = torch.zeros(inp.shape[0], dtype=torch.long, device=dev)

    available = [p for p in router_paths if Path(p).exists()]
    if not available:
        print(f"  [!] no router checkpoints found for {label}, skipping")
        return
    ahat = np.mean([_predict(_load_net(p, dev), inp, prd, pid)
                    for p in available], axis=0)
    r = float(np.corrcoef(A, ahat)[0, 1])
    print(f"  [{label}] n={len(A)}  seeds={len(available)}  Pearson r={r:.4f}")

    lim = float(np.quantile(np.abs(np.concatenate([ahat, A])), 0.99))
    ax.scatter(ahat, A, s=5, alpha=0.25, edgecolors="none", color="tab:red", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=1.0, ls="--", zorder=3, label="$y=x$")
    ax.axhline(0, color="0.6", lw=0.7, ls=":", zorder=2)
    ax.axvline(0, color="0.6", lw=0.7, ls=":", zorder=2)
    ticks = np.linspace(-round(lim, 1), round(lim, 1), 5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Predicted advantage $\hat{A}$")
    ax.set_ylabel(r"True advantage $A = L_\mathrm{base} - L_\mathrm{super}$")
    ax.set_title(label, fontsize=9)
    ax.text(0.04, 0.96, f"Pearson $r$ = {r:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9))
    ax.grid(alpha=0.20, ls="--")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_DEFAULTS), default=None,
                    help="single dataset; omit to run all three")
    args = ap.parse_args()

    plt.rcParams.update(RCPARAMS)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    datasets = ([args.dataset] if args.dataset
                else list(DATASET_DEFAULTS.keys()))

    for ds in datasets:
        cfg = DATASET_DEFAULTS[ds]
        if not Path(cfg["cache"]).exists():
            print(f"  [skip] cache not found: {cfg['cache']}")
            continue
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        draw_panel(ax, cfg["label"], cfg["cache"], cfg["routers"], dev)
        Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(pad=0.4)
        fig.savefig(cfg["out"], bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  [*] -> {cfg['out']}")


if __name__ == "__main__":
    main()
