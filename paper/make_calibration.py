"""Calibration figure: predicted advantage (A-hat) vs. true loss gap (A).

Loads a trained TinyConv router and the validation cache, runs the router to get
A-hat for every val image (previous-action = base), and scatter-plots it against
the true advantage A = L_base - L_super. Reports Pearson r and Spearman rho.
IEEE-style compact panel, vector PDF -> paper/_extracted/fig_calibration.pdf.
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

BASE = ROOT / "method02_advantage_regress_tinyConv/outputs/kitti"
VAL = BASE / "cache_val_g2.pt"
SEEDS = list(range(5))
OUT = Path(__file__).resolve().parent / "_extracted" / "fig_calibration.pdf"

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


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    c = torch.load(VAL, map_location=dev, weights_only=False)
    inp = c["input_base"].to(dev).float()
    A = (c["loss_base"].view(-1) - c["loss_super"].view(-1)).cpu().numpy()
    pid = torch.zeros(inp.shape[0], dtype=torch.long, device=dev)

    # ensemble of seeds reduces per-model noise in the prediction
    ah = np.mean([predict(BASE / f"ablation/policy_input_g2_s{s}.pt", inp, pid, dev)
                  for s in SEEDS], axis=0)

    r = np.corrcoef(A, ah)[0, 1]
    from scipy.stats import spearmanr
    rho = spearmanr(A, ah).correlation

    # binned trend: group by predicted-advantage decile, plot mean true A +/- SEM.
    nb = 10
    order = np.argsort(ah)
    bins = np.array_split(order, nb)
    bx = np.array([ah[b].mean() for b in bins])
    by = np.array([A[b].mean() for b in bins])
    be = np.array([A[b].std() / np.sqrt(len(b)) for b in bins])

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.scatter(ah, A, s=5, alpha=0.18, edgecolors="none", color="tab:red")
    ax.errorbar(bx, by, yerr=be, fmt="o-", color="black", ms=4, lw=1.6, capsize=2,
                label="decile mean $\\pm$ SEM", zorder=5)
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel(r"predicted advantage $\hat{A}$", fontsize=9)
    ax.set_ylabel("true loss gap $A=L_{base}-L_{super}$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.04, 0.96, f"Pearson $r$ = {r:.3f}\nSpearman $\\rho$ = {rho:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"[*] ensemble r={r:.4f} rho={rho:.4f} n={len(A)} -> {OUT}")


if __name__ == "__main__":
    main()
