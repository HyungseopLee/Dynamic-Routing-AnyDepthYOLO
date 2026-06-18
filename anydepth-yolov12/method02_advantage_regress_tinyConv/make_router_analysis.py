"""Router behavior analysis.

(1) Ranking reliability: bin validation images by predicted advantage A-hat into
    deciles and plot the MEAN true advantage A=L_base-L_super per bin. A monotone
    increasing curve shows the router correctly ORDERS frames by how much the deep
    path helps -- the property that matters for threshold routing -- even though the
    per-image scatter is noisy. Reports Pearson and Spearman.

    python -m method02_advantage_regress_tinyConv.make_router_analysis \
        --out_dir ../paper/_extracted
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork  # noqa


def load_router(path, cache, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 64), feat=a.get("feat", "both"),
                        norm=a.get("norm", "batch")).to(device)
    net.eval()
    # build lazy layers with the cache's actual feature dims (KITTI 768 vs BDD 1536)
    ci = cache["input_base"].shape[1]
    g = cache["input_base"].shape[-1]
    pred = cache.get("pred_base")
    cp = pred.shape[1] if pred is not None else 640
    with torch.no_grad():
        net(torch.zeros(2, ci, g, g, device=device),
            torch.zeros(2, cp, g, g, device=device),
            torch.zeros(2, dtype=torch.long, device=device))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, a.get("feat", "both")


@torch.no_grad()
def predict(net, cache, feat, device):
    A = (cache["loss_base"].view(-1) - cache["loss_super"].view(-1)).cpu().numpy()
    pid = torch.zeros(cache["input_base"].shape[0], dtype=torch.long, device=device)
    pred = cache.get("pred_base") if feat in ("both", "pred") else None
    ah = net.logit(cache["input_base"].to(device),
                   pred.to(device) if pred is not None else None, pid).view(-1).cpu().numpy()
    return ah, A


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-8))


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT.parent / "paper/_extracted"))
    ap.add_argument("--nbins", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    B = "method02_advantage_regress_tinyConv/outputs"
    datasets = [
        ("KITTI", f"{B}/kitti/policy.pt", f"{B}/kitti/cache_val_g2.pt"),
        ("BDD100K", f"{B}/bdd100k/policy_scenario_s0.pt", f"{B}/bdd100k/cache_val_g2.pt"),
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # match the main-result figures: Times/serif text, no in-figure title (the
    # (a)/(b) dataset labels are added below in LaTeX, exactly like Fig.~main).
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--"})
    tag = {"KITTI": "kitti", "BDD100K": "bdd"}
    advs = {}
    for name, pol, cache_p in datasets:
        cache = torch.load(cache_p, map_location=dev, weights_only=False)
        net, feat = load_router(pol, cache, dev)
        ah, A = predict(net, cache, feat, dev)
        advs[name] = A
        r, rho = pearson(ah, A), spearman(ah, A)
        order = np.argsort(ah)
        bins = np.array_split(order, args.nbins)
        ym = [A[b].mean() for b in bins]
        ys = [A[b].std() / np.sqrt(len(b)) for b in bins]
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        # discrete deciles: markers only, no connecting line (x is categorical)
        ax.errorbar(range(1, args.nbins + 1), ym, yerr=ys, fmt="o", color="tab:red",
                    markersize=5, capsize=2, elinewidth=1.2)
        ax.set_xlabel(r"predicted advantage $\hat{A}$ (decile)", fontsize=10)
        ax.set_ylabel(r"mean true advantage $A$", fontsize=10)
        ax.set_xticks(range(1, args.nbins + 1))
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        out = Path(args.out_dir) / f"fig_calibration_{tag[name]}.pdf"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        ratio = ym[-1] / ym[0] if ym[0] > 0 else float("nan")
        print(f"{name}: n={len(A)} feat={feat} Pearson={r:.3f} Spearman={rho:.3f} "
              f"ratio={ratio:.1f}x -> {out}")

    # advantage distribution (same serif style); x = true advantage A
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    colors = {"KITTI": "tab:red", "BDD100K": "tab:blue"}
    for name, A in advs.items():
        ax.hist(A, bins=80, range=(-0.6, 0.7), density=True, histtype="stepfilled",
                color=colors[name], alpha=0.35, edgecolor=colors[name], linewidth=1.2,
                label=rf"{name} ($\sigma{{=}}{A.std():.2f}$)")
    ax.axvline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel(r"true advantage $A=L_{\mathrm{base}}-L_{\mathrm{super}}$", fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = Path(args.out_dir) / "fig_advantage_dist.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[*] -> {out}")


if __name__ == "__main__":
    main()
