"""Router spatial & temporal visualization.

Two experiments:
  1. Spatial Grid Activation Map  — 2x2 cell ablation heatmap overlaid on image
  2. Temporal Routing Sequence    — per-frame super/base decision timeline for a clip

Usage:
    python -m method_advantage_regress.analysis.visualize_router_spatial_temporal \
        --cache  method_advantage_regress/outputs/bdd100k/cache_val_both.pt \
        --router method_advantage_regress/outputs/bdd100k/router_both_0.pt \
        --images /media/data/bdd100k_yolo/val/images \
        --out    method_advantage_regress/outputs/figures/router_spatial_temporal
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def load_router(router_path: str):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from method_advantage_regress.router.router_net import RouterNetwork, GapMlpNet
    ckpt = torch.load(router_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    is_gap = not any(k.endswith("weight") and v.dim() == 4 for k, v in sd.items())
    if is_gap:
        group_dim  = sd["input_proj.fc.weight"].shape[0]
        hidden_dim = sd["head.0.weight"].shape[0]
        net = GapMlpNet(group_dim=group_dim, hidden_dim=hidden_dim)
    else:
        group_dim  = sd["input_proj.depth.weight"].shape[0]
        hidden_dim = sd["head.0.weight"].shape[0]
        net = RouterNetwork(group_dim=group_dim, hidden_dim=hidden_dim)
    net.load_state_dict(sd)
    net.eval()
    return net


def predict_one(net, xi: torch.Tensor, xp: torch.Tensor) -> float:
    """xi, xp: [1, C, 2, 2]. Returns scalar logit."""
    from method_advantage_regress.router.router_net import GapMlpNet
    pid = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        if isinstance(net, GapMlpNet):
            logit = net.logit(xi.mean(dim=(2, 3)), xp.mean(dim=(2, 3)), pid)
        else:
            logit = net.logit(xi, xp, pid)
    return logit.item()


def cell_attribution_gradient(net, xi: torch.Tensor, xp: torch.Tensor) -> np.ndarray:
    """Gradient × Input attribution (saliency) per 2×2 cell.

    Uses |grad * input| so the result varies per frame (not just model weights).
    Returns [2, 2] array, always non-negative: higher = this cell is
    both sensitive AND active in this specific frame.
    """
    from method_advantage_regress.router.router_net import GapMlpNet
    pid = torch.zeros(1, dtype=torch.long)
    xi_g = xi.clone().requires_grad_(True)
    xp_g = xp.clone().requires_grad_(True)
    if isinstance(net, GapMlpNet):
        logit = net.logit(xi_g.mean(dim=(2, 3)), xp_g.mean(dim=(2, 3)), pid)
    else:
        logit = net.logit(xi_g, xp_g, pid)
    logit.backward()
    # gradient × input per cell: [B, C, 2, 2] → [2, 2]
    sal_xi = (xi_g.grad * xi_g).abs().sum(dim=(0, 1))
    sal_xp = (xp_g.grad * xp_g).abs().sum(dim=(0, 1))
    return (sal_xi + sal_xp).detach().numpy()


def cell_attribution(net, xi: torch.Tensor, xp: torch.Tensor) -> np.ndarray:
    """Ablation attribution: contribution of each 2x2 cell to Â.

    Returns [2, 2] array: positive = this cell raises the prediction.
    Both input and pred features are zeroed together for each cell.
    """
    base_val = predict_one(net, xi, xp)
    contrib = np.zeros((2, 2))
    for r in range(2):
        for c in range(2):
            xi_m = xi.clone()
            xp_m = xp.clone()
            xi_m[:, :, r, c] = 0.0
            xp_m[:, :, r, c] = 0.0
            contrib[r, c] = base_val - predict_one(net, xi_m, xp_m)
    return contrib


# ── Experiment 1: Spatial Grid Activation Map ─────────────────────────────────

def _overlay_heatmap(ax, img: np.ndarray, grad_map: np.ndarray,
                     a_true: float, a_hat: float):
    """Overlay normalised gradient heatmap on image."""
    H, W = img.shape[:2]
    ax.imshow(img)
    vmax = grad_map.max() + 1e-9
    cmap = plt.get_cmap("hot")
    for r in range(2):
        for c in range(2):
            val = grad_map[r, c]
            alpha = 0.15 + 0.55 * (val / vmax)
            color = cmap(val / vmax)
            ax.add_patch(patches.Rectangle(
                (c * W / 2, r * H / 2), W / 2, H / 2,
                linewidth=0, facecolor=color, alpha=alpha))
            ax.add_patch(patches.Rectangle(
                (c * W / 2, r * H / 2), W / 2, H / 2,
                linewidth=1.5, edgecolor="white", facecolor="none"))
            rank = int((val / vmax) * 100)
            ax.text((c + 0.5) * W / 2, (r + 0.5) * H / 2,
                    f"{rank}%", color="white", fontsize=10,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))
    ax.set_title(f"True A={a_true:.3f}  Â={a_hat:.3f}", fontsize=10)
    ax.axis("off")


def plot_spatial_grid_map(net, cache, image_dir: Path, out: Path, topk: int = 9):
    """Gradient-based 2x2 spatial attention on top-K high-Â frames."""
    adv_true    = (cache["loss_base"] - cache["loss_super"]).numpy()
    input_feats = cache["input_base"]   # [N, 768, 2, 2]
    pred_feats  = cache["pred_base"]    # [N, 640, 2, 2]
    im_files    = cache["im_file"]

    # --- compute Â for all frames (batch) ---
    from method_advantage_regress.analysis.analyze_router_behavior import predict_advantage
    adv_pred = predict_advantage(net, input_feats, pred_feats)

    # pick top-K by Â (high confidence SUPER → most spatial signal)
    top_idx = np.argsort(adv_pred)[-topk:][::-1]

    n_cols = 3
    n_rows = (topk + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.array(axes).reshape(-1)

    for plot_i, frame_i in enumerate(top_idx):
        ax = axes[plot_i]
        xi = input_feats[frame_i:frame_i+1]
        xp = pred_feats[frame_i:frame_i+1]
        grad_map = cell_attribution_gradient(net, xi, xp)   # [2, 2] >= 0

        fname = os.path.basename(im_files[frame_i])
        img_path = image_dir / fname
        img = np.array(Image.open(img_path).convert("RGB")) if img_path.exists() \
              else np.zeros((720, 1280, 3), dtype=np.uint8)

        _overlay_heatmap(ax, img, grad_map, adv_true[frame_i], adv_pred[frame_i])

    sm = ScalarMappable(cmap="hot", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:topk], shrink=0.4,
                 label="Normalised gradient attention\n(bright=high influence on Â)")
    for ax in axes[topk:]:
        ax.set_visible(False)

    fig.suptitle("Spatial Grid Activation Map: Top-K High-Confidence SUPER Frames\n"
                 "Gradient attribution shows which 2×2 region influences the router most",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out / "7_spatial_grid_activation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    _plot_dominant_cell_stats(net, cache, adv_pred, out)


def _plot_dominant_cell_stats(net, cache, adv_pred: np.ndarray, out: Path):
    """Aggregate gradient attention: which cell is most active for high-Â frames?"""
    input_feats = cache["input_base"]
    pred_feats  = cache["pred_base"]

    # top-25% by Â
    thresh = np.percentile(adv_pred, 75)
    indices = np.where(adv_pred >= thresh)[0]

    dom_counts = np.zeros((2, 2))
    grad_all   = np.zeros((2, 2))

    for i in indices:
        xi = input_feats[i:i+1]
        xp = pred_feats[i:i+1]
        g = cell_attribution_gradient(net, xi, xp)
        dom = np.unravel_index(np.argmax(g), g.shape)
        dom_counts[dom] += 1
        grad_all += g / (g.sum() + 1e-9)   # normalise before averaging

    mean_grad = grad_all / len(indices)
    cell_names = [["Top-Left\n(TL)", "Top-Right\n(TR)"],
                  ["Bot-Left\n(BL)", "Bot-Right\n(BR)"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title, cmap_name, fmt in [
        (axes[0], dom_counts / dom_counts.sum(),
         "Dominant Cell Frequency\n(top-25% Â frames)", "YlOrRd", ".1%"),
        (axes[1], mean_grad,
         "Mean Normalised Gradient\n(top-25% Â frames)", "YlOrRd", ".4f"),
    ]:
        im = ax.imshow(data, cmap=cmap_name, vmin=0)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cell_names[r][c]}\n{data[r,c]:{fmt}}",
                        ha="center", va="center", fontsize=10,
                        color="white" if data[r, c] > data.max() * 0.55 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Left", "Right"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Top", "Bottom"])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Which 2×2 Grid Cell Drives High-Advantage Predictions?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out / "8_dominant_cell_stats.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Experiment 2: Temporal Routing Sequence ───────────────────────────────────

def plot_temporal_routing(net, cache, image_dir: Path, out: Path,
                          min_frames: int = 8, n_clips: int = 3):
    """Timeline of router confidence (Â) on consecutive driving frames.

    Since Â > 0 for all frames (router always leans SUPER), we show the
    continuous confidence signal and use the 50th-percentile as a 'high/low'
    visual threshold, mirroring how the PI budget controller varies τ.
    """
    from method_advantage_regress.analysis.analyze_router_behavior import predict_advantage
    adv_true    = (cache["loss_base"] - cache["loss_super"]).numpy()
    input_feats = cache["input_base"]
    pred_feats  = cache["pred_base"]
    im_files    = cache["im_file"]

    # pre-compute Â for all frames
    adv_pred = predict_advantage(net, input_feats, pred_feats)
    global_med = np.median(adv_pred)

    # group frames by video id (BDD100K: videoId-frameId.jpg)
    video_frames = defaultdict(list)
    for idx, f in enumerate(im_files):
        stem_name = os.path.basename(f)
        vid = stem_name.split("-")[0]
        video_frames[vid].append((stem_name, idx))

    for vid in video_frames:
        video_frames[vid].sort(key=lambda x: x[0])

    # pick clips: prefer large Â range over the clip (most dynamic)
    clips = [(vid, frames) for vid, frames in video_frames.items()
             if len(frames) >= min_frames]

    def clip_score(frames):
        idxs = [f[1] for f in frames]
        a_hat = adv_pred[idxs]
        return a_hat.max() - a_hat.min()   # max Â swing

    clips.sort(key=lambda x: clip_score(x[1]), reverse=True)
    selected = clips[:n_clips]

    for clip_i, (vid, frames) in enumerate(selected):
        idxs  = [f[1] for f in frames]
        n     = len(idxs)
        a_true = adv_true[idxs]
        a_hat  = adv_pred[idxs]

        # "high confidence" = Â above global median
        is_high = a_hat >= global_med

        # thumbnails: choose frames at Â extremes + evenly spaced
        n_thumb = min(8, n)
        thumb_pos = np.round(np.linspace(0, n - 1, n_thumb)).astype(int)

        fig = plt.figure(figsize=(16, 7.5))
        gs = plt.GridSpec(3, n_thumb, figure=fig,
                          height_ratios=[2.5, 0.12, 1.8], hspace=0.06)

        # ── Row 0: thumbnails ──────────────────────────────────────────────
        for ti, fi in enumerate(thumb_pos):
            ax_img = fig.add_subplot(gs[0, ti])
            fname = os.path.basename(im_files[idxs[fi]])
            img_path = image_dir / fname
            if img_path.exists():
                img = Image.open(img_path).convert("RGB").resize((320, 180))
                ax_img.imshow(np.array(img))
            else:
                ax_img.set_facecolor("#333333")

            # border: orange=HIGH confidence, teal=LOW confidence
            color = "#e63946" if is_high[fi] else "#2196F3"
            label = f"HIGH\nÂ={a_hat[fi]:.3f}" if is_high[fi] else f"LOW\nÂ={a_hat[fi]:.3f}"
            for spine in ax_img.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(5)
            ax_img.set_xticks([]); ax_img.set_yticks([])
            ax_img.set_xlabel(f"t={fi+1}\n{label}",
                              color=color, fontsize=7.5, fontweight="bold")

        # ── Row 1: Â confidence bar ────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[1, :])
        norm_hat = (a_hat - a_hat.min()) / (a_hat.max() - a_hat.min() + 1e-9)
        ax_bar.imshow(norm_hat.reshape(1, -1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax_bar.set_xticks([]); ax_bar.set_yticks([])
        ax_bar.set_ylabel("Â level", fontsize=7, rotation=0, labelpad=35, va="center")
        for fi in thumb_pos:
            ax_bar.axvline(fi, color="white", linewidth=1.5, alpha=0.7)

        # ── Row 2: advantage plot ──────────────────────────────────────────
        ax_line = fig.add_subplot(gs[2, :])
        x = np.arange(n)
        ax_line.plot(x, a_true, color="crimson", linewidth=2, label="True A", zorder=3)
        ax_line.plot(x, a_hat,  color="steelblue", linewidth=1.5,
                     linestyle="--", label="Predicted Â (router confidence)", zorder=3)
        ax_line.axhline(global_med, color="darkorange", linewidth=1.2,
                        linestyle=":", label=f"Global median Â={global_med:.3f}")
        ax_line.fill_between(x, global_med, a_hat,
                             where=a_hat >= global_med, alpha=0.15, color="crimson",
                             label="HIGH confidence SUPER")
        ax_line.fill_between(x, global_med, a_hat,
                             where=a_hat < global_med,  alpha=0.15, color="steelblue",
                             label="LOW confidence SUPER")

        for fi in thumb_pos:
            ax_line.axvline(fi, color="gray", linewidth=1, alpha=0.5, linestyle=":")

        ax_line.set_xlabel("Frame index (temporal order)", fontsize=10)
        ax_line.set_ylabel("Advantage / Â", fontsize=10)
        ax_line.legend(fontsize=8, loc="upper right", ncol=2)
        ax_line.grid(alpha=0.3)
        ax_line.set_xlim(0, n - 1)

        fig.suptitle(
            f"Temporal Routing Confidence — Video clip '{vid}' ({n} frames)\n"
            f"Â range: [{a_hat.min():.3f}, {a_hat.max():.3f}]  |  "
            f"True A range: [{a_true.min():.3f}, {a_true.max():.3f}]",
            fontsize=12, fontweight="bold"
        )

        out_path = out / f"9_temporal_routing_clip{clip_i+1}_{vid}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


# ── Combined figure for paper ─────────────────────────────────────────────────

def plot_combined_paper_figure(net, cache, image_dir: Path, out: Path):
    """Single figure combining spatial heatmap + temporal routing for one frame."""
    adv_true = (cache["loss_base"] - cache["loss_super"]).numpy()
    input_feats = cache["input_base"]
    pred_feats  = cache["pred_base"]
    im_files    = cache["im_file"]

    # find the single frame with highest true A that has an actual image
    top_idx = np.argsort(adv_true)[::-1]
    best_i = None
    for i in top_idx:
        fname = os.path.basename(im_files[i])
        if (image_dir / fname).exists():
            best_i = i
            break
    if best_i is None:
        print("No images found for combined figure, skipping.")
        return

    from method_advantage_regress.analysis.analyze_router_behavior import predict_advantage
    adv_pred = predict_advantage(net, input_feats, pred_feats)

    # pick frame with highest Â that has an image
    top_idx = np.argsort(adv_pred)[::-1]
    best_i = None
    for i in top_idx:
        fname = os.path.basename(im_files[i])
        if (image_dir / fname).exists():
            best_i = i
            break
    if best_i is None:
        print("No images found for combined figure, skipping.")
        return

    xi = input_feats[best_i:best_i+1]
    xp = pred_feats[best_i:best_i+1]
    grad_map = cell_attribution_gradient(net, xi, xp)   # [2, 2] >= 0
    a_true = (cache["loss_base"] - cache["loss_super"]).numpy()[best_i]
    a_hat  = adv_pred[best_i]

    fname = os.path.basename(im_files[best_i])
    img = np.array(Image.open(image_dir / fname).convert("RGB"))
    H, W = img.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: raw image with heatmap
    ax = axes[0]
    _overlay_heatmap(ax, img, grad_map, a_true, a_hat)
    sm = ScalarMappable(cmap="hot", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Normalised gradient attention")

    # right: 2x2 matrix (normalised)
    ax2 = axes[1]
    norm_g = grad_map / (grad_map.sum() + 1e-9)
    im = ax2.imshow(norm_g, cmap="YlOrRd", vmin=0)
    cell_names = [["Top-Left", "Top-Right"], ["Bot-Left", "Bot-Right"]]
    for r in range(2):
        for c in range(2):
            ax2.text(c, r, f"{cell_names[r][c]}\n{norm_g[r,c]:.1%}",
                     ha="center", va="center", fontsize=12,
                     color="white" if norm_g[r, c] > norm_g.max() * 0.55 else "black",
                     fontweight="bold")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Left", "Right"], fontsize=10)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Top", "Bottom"], fontsize=10)
    ax2.set_title("2×2 Cell Attention Share\n(Gradient attribution, normalised)",
                  fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax2, shrink=0.6)

    plt.suptitle("Router Interpretability: Spatial Grid Activation Map",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out / "10_paper_spatial_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache",  default="method_advantage_regress/outputs/bdd100k/cache_val_both.pt")
    p.add_argument("--router", default="method_advantage_regress/outputs/bdd100k/router_both_0.pt")
    p.add_argument("--images", default="/media/data/bdd100k_yolo/val/images")
    p.add_argument("--out",    default="method_advantage_regress/outputs/figures/router_spatial_temporal")
    p.add_argument("--topk",   type=int, default=9, help="frames for spatial map")
    p.add_argument("--clips",  type=int, default=3, help="number of temporal clips")
    p.add_argument("--skip-spatial",   action="store_true")
    p.add_argument("--skip-temporal",  action="store_true")
    p.add_argument("--skip-combined",  action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.images)

    print("Loading cache and router...")
    cache = torch.load(args.cache, map_location="cpu")
    net   = load_router(args.router)

    if not args.skip_spatial:
        print("\n=== Experiment 1: Spatial Grid Activation Map ===")
        plot_spatial_grid_map(net, cache, image_dir, out, topk=args.topk)

    if not args.skip_temporal:
        print("\n=== Experiment 2: Temporal Routing Sequence ===")
        plot_temporal_routing(net, cache, image_dir, out, n_clips=args.clips)

    if not args.skip_combined:
        print("\n=== Combined Paper Figure ===")
        plot_combined_paper_figure(net, cache, image_dir, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
