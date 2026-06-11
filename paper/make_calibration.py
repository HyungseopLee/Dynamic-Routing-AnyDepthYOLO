"""Calibration figure: predicted advantage (A-hat) vs. true loss gap (A).

For each dataset, loads the trained TinyConv routers and the validation cache,
runs the routers (previous-action = base) to get A-hat for every val image, and
scatter-plots it against the true advantage A = L_base - L_super. Reports Pearson r.
x and y use an identical scale/ticks with a y=x reference line. Two panels:
(a) KITTI, (b) BDD100K. IEEE-style, vector PDF -> paper/_extracted/fig_calibration.pdf.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent / "anydepth-yolov12"
sys.path.insert(0, str(ROOT))
from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork  # noqa: E402

OUT = Path(__file__).resolve().parent / "_extracted" / "fig_calibration.pdf"
M2 = ROOT / "method02_advantage_regress_tinyConv/outputs"
SEEDS = list(range(5))

# (label, val-cache, policy-checkpoint glob template over seeds)
DATASETS = [
    ("KITTI",   M2 / "kitti/cache_val_g2.pt",   M2 / "kitti/ablation/policy_input_g2_s{s}.pt"),
    ("BDD100K", M2 / "bdd100k/cache_val_g2.pt",  M2 / "bdd100k/ablation/policy_input_g2_s{s}.pt"),
]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def predict(ckpt, inp, pid, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 128), feat=a.get("feat", "input"),
                        norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(dev)
    with torch.no_grad():
        net(inp[:2], None, pid[:2])          # materialize lazy layers
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        return net.logit(inp, None, pid).view(-1).cpu().numpy()


def panel(ax, letter, label, val, glob, dev):
    c = torch.load(val, map_location=dev, weights_only=False)
    inp = c["input_base"].to(dev).float()
    A = (c["loss_base"].view(-1) - c["loss_super"].view(-1)).cpu().numpy()
    pid = torch.zeros(inp.shape[0], dtype=torch.long, device=dev)
    ah = np.mean([predict(Path(str(glob).format(s=s)), inp, pid, dev) for s in SEEDS], axis=0)
    r = float(np.corrcoef(A, ah)[0, 1])
    print(f"[{label}] ensemble r={r:.4f} n={len(A)}")

    ax.scatter(ah, A, s=6, alpha=0.30, edgecolors="none", color="tab:red")
    lim = float(np.quantile(np.abs(np.concatenate([ah, A])), 0.99))
    ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=1.0, ls="--", zorder=1)
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ticks = np.linspace(-round(lim, 1), round(lim, 1), 5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"predicted advantage $\hat{A}$", fontsize=9)
    ax.set_ylabel(r"true gap $A=L_{base}-L_{super}$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"({letter}) {label}", fontsize=9)
    ax.text(0.04, 0.96, f"Pearson $r$ = {r:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.grid(alpha=0.25, ls="--")


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    avail = [(lbl, val, glob) for (lbl, val, glob) in DATASETS
             if Path(val).exists() and Path(str(glob).format(s=0)).exists()]
    missing = [lbl for (lbl, val, glob) in DATASETS if (lbl, val, glob) not in avail]
    if missing:
        print(f"[!] not yet available, rendering {[a[0] for a in avail]} only: missing {missing}")
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.3), squeeze=False)
    for i, (label, val, glob) in enumerate(avail):
        panel(axes[0][i], "ab"[i], label, val, glob, dev)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"[*] -> {OUT}")


if __name__ == "__main__":
    main()
