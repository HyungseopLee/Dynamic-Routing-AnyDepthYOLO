"""Visualize whether the router catches base-better (A<0) frames on BDD100K.

(a) scatter: true advantage A vs predicted A-hat, with the four sign quadrants.
    The bottom-left quadrant (A<0 & A-hat<0) is the ONLY place the router correctly
    keeps a base-better frame on BASE. We show it is nearly empty.
(b) for true-A bins, fraction the router sends to SUPER (decision = A-hat>tau);
    a working router would drop toward 0 on the left (A<0). We show it stays high.
(c) ROC for discriminating sign(A): the curve hugs the diagonal (AUROC ~0.5).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})


@torch.no_grad()
def predict(cache, policy, dev):
    c = torch.load(cache, map_location="cpu", weights_only=False)
    A = (c["loss_base"] - c["loss_super"]).numpy()
    ck = torch.load(policy, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 64), feat=a.get("feat", "input"),
                        norm=a.get("norm", "batch"), dropout=0.0).to(dev).eval()
    inp = c["input_base"].to(dev); pid = torch.zeros(inp.shape[0], dtype=torch.long, device=dev)
    net(inp[:2], None, pid[:2]); net.load_state_dict(ck["state_dict"]); net.eval()
    H = net.logit(inp, None, pid).view(-1).cpu().numpy()
    return A, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="method02_advantage_regress_tinyConv/outputs/bdd100k/cache_val_g2.pt")
    ap.add_argument("--policy", default="method02_advantage_regress_tinyConv/outputs/bdd100k/policy_scenario_s0.pt")
    ap.add_argument("--out", default="method02_advantage_regress_tinyConv/outputs/bdd100k/sign_viz.pdf")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    A, H = predict(args.cache, args.policy, dev)
    # decision threshold: use the median A-hat so ~50% route each way (fair view of
    # discrimination, not the offset). Also report tau=0 line.
    tau = float(np.median(H))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    # ---- (a) scatter A vs A-hat ----
    a0 = ax[0]
    neg, pos = A < 0, A >= 0
    a0.scatter(A[pos], H[pos], s=6, c="#1f77b4", alpha=0.25, label="$A>0$ (super better)")
    a0.scatter(A[neg], H[neg], s=6, c="#d62728", alpha=0.35, label="$A<0$ (base better)")
    a0.axvline(0, color="k", lw=0.8); a0.axhline(tau, color="k", ls="--", lw=0.8)
    a0.text(0.02, 0.96, f"router keeps BASE here\n(want red, got "
            f"{((H < tau) & neg).sum()})", transform=a0.transAxes, va="top",
            fontsize=8, color="#d62728")
    rho = np.corrcoef(np.argsort(np.argsort(A)), np.argsort(np.argsort(H)))[0, 1]
    a0.set_xlabel("true advantage $A=L_{base}-L_{super}$")
    a0.set_ylabel(r"router prediction $\hat{A}$ (logit)")
    a0.set_title(f"(a) prediction vs truth  ($\\rho$={rho:.2f})")
    a0.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # ---- (b) P(route SUPER) vs true-A bin ----
    a1 = ax[1]
    bins = np.quantile(A, np.linspace(0, 1, 13))
    bins = np.unique(bins)
    idx = np.digitize(A, bins[1:-1])
    centers, frac, base_share = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        centers.append(A[m].mean())
        frac.append((H[m] > tau).mean())
        base_share.append((A[m] < 0).mean())
    a1.plot(centers, np.array(frac) * 100, "o-", color="#1f77b4", label="router $\\to$SUPER")
    a1.plot(centers, [0 if c >= 0 else 100 for c in centers], "k--", lw=1,
            label="oracle (BASE if $A<0$)")
    a1.axvline(0, color="grey", lw=0.8)
    a1.set_xlabel("true advantage $A$ (binned)")
    a1.set_ylabel("% routed to SUPER")
    a1.set_title("(b) routing vs true advantage")
    a1.legend(fontsize=8, loc="lower right")

    # ---- (c) ROC for sign(A) ----
    a2 = ax[2]
    y = (A > 0).astype(int)
    fpr, tpr, _ = roc_curve(y, H)
    auc = roc_auc_score(y, H)
    a2.plot(fpr, tpr, color="#1f77b4", label=f"router (AUROC={auc:.2f})")
    a2.plot([0, 1], [0, 1], "k--", lw=1, label="chance (0.50)")
    a2.set_xlabel("false positive rate"); a2.set_ylabel("true positive rate")
    a2.set_title("(c) sign discrimination ROC")
    a2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[*] saved {args.out}  (AUROC={auc:.3f}, rho={rho:.3f}, "
          f"A<0 kept-on-base={((H < tau) & neg).sum()}/{neg.sum()})")


if __name__ == "__main__":
    main()
