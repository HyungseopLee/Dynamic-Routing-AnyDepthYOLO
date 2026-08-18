"""Path-consistency analysis: do base-path and super-path features give the same router score?

For each frame in the val set, computes router logit twice:
  - inp = input_base  (context from previous BASE frame)
  - inp = input_super (context from previous SUPER frame)

Produces a 3-panel scatter figure comparing p=0.0 / p=0.5 / p=1.0 training conditions.
The key message: only p=0.5 achieves both high correlation AND low MAE (scores lie on y=x).

Usage:
    python -m analysis.plot_path_consistency \
        --cache results/step2_router/cache/bdd100k/cache_val_both.pt \
        --out   results/figures/fig_path_consistency_bdd.pdf
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.router_net import GapMlpNet, RouterNetwork


# ── router loading (mirrors eval_video.load_router) ─────────────────────────
def load_router(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    a  = ckpt.get("args", {})
    is_gap = not any(k.endswith("weight") and v.dim() == 4 for k, v in sd.items())

    if is_gap and any(k.startswith("input_proj.0.") for k in sd):
        remap = {}
        for k, v in sd.items():
            nk = k.replace("input_proj.0.", "input_proj.norm.") \
                   .replace("input_proj.1.", "input_proj.fc.")  \
                   .replace("pred_proj.0.",  "pred_proj.norm.") \
                   .replace("pred_proj.1.",  "pred_proj.fc.")
            remap[nk] = v
        sd = remap

    cls = GapMlpNet if is_gap else RouterNetwork
    net = cls(group_dim=a.get("group_dim", 64),
              path_dim=a.get("path_dim", 8),
              hidden_dim=a.get("hidden", 64),
              feat=a.get("feat", "both"),
              norm=a.get("norm", "batch"),
              dropout=a.get("dropout", 0.0))
    net.load_state_dict(sd, strict=True)
    net.eval().to(device)
    return net, a.get("feat", "both"), is_gap


@torch.no_grad()
def score_all(net, cache, feat, is_gap, device, batch=512):
    N = cache["input_base"].shape[0]
    sb, ss = [], []
    for i in range(0, N, batch):
        ib = cache["input_base"][i:i+batch].to(device, dtype=torch.float32)
        is_ = cache["input_super"][i:i+batch].to(device, dtype=torch.float32)
        pb = cache["pred_base"][i:i+batch].to(device, dtype=torch.float32)
        ps = cache["pred_super"][i:i+batch].to(device, dtype=torch.float32)

        if is_gap:
            ib = ib.mean(dim=(2, 3));  is_ = is_.mean(dim=(2, 3))
            pb = pb.mean(dim=(2, 3));  ps  = ps.mean(dim=(2, 3))

        pid0 = torch.zeros(ib.shape[0], dtype=torch.long, device=device)
        pid1 = torch.ones(ib.shape[0],  dtype=torch.long, device=device)

        if feat == "input":
            sb.append(net.logit(ib,   None, pid0).cpu())
            ss.append(net.logit(is_,  None, pid1).cpu())
        elif feat == "pred":
            sb.append(net.logit(None, pb,   pid0).cpu())
            ss.append(net.logit(None, ps,   pid1).cpu())
        else:  # both
            sb.append(net.logit(ib,   pb,   pid0).cpu())
            ss.append(net.logit(is_,  ps,   pid1).cpu())

    return torch.cat(sb).numpy().ravel(), torch.cat(ss).numpy().ravel()


def collect_seeds(ckpt_paths, cache, device):
    """Return pooled (sb, ss) arrays over all seeds."""
    all_sb, all_ss = [], []
    for p in ckpt_paths:
        net, feat, is_gap = load_router(p, device)
        sb, ss = score_all(net, cache, feat, is_gap, device)
        all_sb.append(sb); all_ss.append(ss)
    return np.concatenate(all_sb), np.concatenate(all_ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",  default="results/step2_router/cache/bdd100k/cache_val_both.pt")
    ap.add_argument("--out",    default="results/figures/fig_path_consistency_bdd.pdf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_pts",  type=int, default=4000,
                    help="points to scatter per panel (subsampled for clarity)")
    args = ap.parse_args()

    BDD = "results/step3_eval/bdd100k"
    GROUPS = [
        ("$p=0.0$",           [f"{BDD}/router_both_prevpp00_s{i}.pt" for i in range(5)]),
        ("$p=0.5$ (ours)",    [f"{BDD}/router_both_{i}.pt"           for i in range(5)]),
        ("$p=1.0$",           [f"{BDD}/router_both_prevpp10_s{i}.pt" for i in range(5)]),
    ]

    cache = torch.load(args.cache, map_location="cpu")
    print(f"[*] val cache: {cache['input_base'].shape[0]} frames")

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    for ax, (label, ckpts) in zip(axes, GROUPS):
        sb, ss = collect_seeds(ckpts, cache, args.device)
        r, _  = pearsonr(sb, ss)
        mae   = float(np.mean(np.abs(sb - ss)))
        print(f"  {label:20s}  r={r:.4f}  MAE={mae:.4f}")

        # subsample for scatter
        idx = np.random.default_rng(0).choice(len(sb), min(args.n_pts, len(sb)), replace=False)
        color = "tab:blue" if "0.5" in label else "tab:gray"
        ax.scatter(sb[idx], ss[idx], s=3, alpha=0.35, color=color, rasterized=True)

        lo = min(sb.min(), ss.min()); hi = max(sb.max(), ss.max())
        ax.plot([lo, hi], [lo, hi], color="tab:red", lw=1.8, ls="--", label="$y=x$")

        is_ours = "0.5" in label
        box_col = "#d4edda" if is_ours else "#f8d7da"
        ax.text(0.05, 0.96,
                f"$r$ = {r:.3f}\nMAE = {mae:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=box_col, alpha=0.85, edgecolor="none"))

        ax.set_title(label, fontsize=14, fontweight="bold" if is_ours else "normal")
        ax.set_xlabel("Score  (prev $=$ base)", fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel("Score  (prev $=$ super)", fontsize=12)
        ax.legend(frameon=False, fontsize=11, loc="lower right")

    fig.suptitle("Path consistency: router score with base vs. super path context  (BDD100K val)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
