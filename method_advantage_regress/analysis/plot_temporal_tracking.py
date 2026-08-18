"""Temporal routing figure where Predicted Â closely tracks True A.

Selects clips with highest Pearson correlation between True A and Â,
then renders a publication-quality figure with:
  - Thumbnail row (colored border: HIGH Â = red, LOW Â = blue)
  - Dual-axis line plot: True A (left axis) and Â (right axis, scaled)
  - Correlation annotation

Usage:
    python -m method_advantage_regress.analysis.plot_temporal_tracking \
        --cache  method_advantage_regress/outputs/bdd100k/cache_val_both.pt \
        --router method_advantage_regress/outputs/bdd100k/router_both_0.pt \
        --images /media/data/bdd100k_yolo/val/images \
        --out    method_advantage_regress/outputs/figures/router_spatial_temporal \
        --clips  c91a84db c07733d0 b75da19e
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image


def load_router(router_path: str):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from method_advantage_regress.router.router_net import RouterNetwork, GapMlpNet
    ckpt = torch.load(router_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    is_gap = not any(k.endswith("weight") and v.dim() == 4 for k, v in sd.items())
    if is_gap:
        net = GapMlpNet(group_dim=sd["input_proj.fc.weight"].shape[0],
                        hidden_dim=sd["head.0.weight"].shape[0])
    else:
        net = RouterNetwork(group_dim=sd["input_proj.depth.weight"].shape[0],
                            hidden_dim=sd["head.0.weight"].shape[0])
    net.load_state_dict(sd)
    net.eval()
    return net


def predict_advantage(net, input_feats, pred_feats):
    from method_advantage_regress.router.router_net import GapMlpNet
    pid = torch.zeros(input_feats.shape[0], dtype=torch.long)
    with torch.no_grad():
        if isinstance(net, GapMlpNet):
            logits = net.logit(input_feats.mean(dim=(2, 3)),
                               pred_feats.mean(dim=(2, 3)), pid)
        else:
            logits = net.logit(input_feats, pred_feats, pid)
    return logits.squeeze(-1).numpy()


def auto_select_clips(adv_true, adv_pred, im_files, n=3, min_frames=5):
    video_frames = defaultdict(list)
    for idx, f in enumerate(im_files):
        vid = os.path.basename(f).split("-")[0]
        video_frames[vid].append((os.path.basename(f), idx))
    for vid in video_frames:
        video_frames[vid].sort(key=lambda x: x[0])

    scored = []
    for vid, frames in video_frames.items():
        if len(frames) < min_frames:
            continue
        idxs = [f[1] for f in frames]
        at = adv_true[idxs]
        ap = adv_pred[idxs]
        if at.std() < 1e-6:
            continue
        corr = float(np.corrcoef(at, ap)[0, 1])
        a_range = float(at.max() - at.min())
        scored.append((corr * a_range * len(frames), vid, frames, corr, a_range))

    scored.sort(reverse=True)
    return [(vid, frames, corr, ar) for _, vid, frames, corr, ar in scored[:n]]


def _frame_x_positions(frames):
    """Extract proportional x positions from BDD100K hex frame IDs.

    BDD100K filenames: videoId-hexFrameId.jpg
    Convert hex IDs to integers, then normalise so x[0]=0, x[-1]=n-1
    preserving the relative temporal gaps between sampled frames.
    """
    hex_ids = [int(f[0].split("-")[1].replace(".jpg", ""), 16) for f in frames]
    ids = np.array(hex_ids, dtype=np.float64)
    ids -= ids[0]
    if ids[-1] > 0:
        ids = ids / ids[-1] * (len(frames) - 1)   # scale to [0, n-1]
    return ids


def plot_clip(vid, frames, idxs, a_true, a_hat, im_files, image_dir, out_dir, clip_i):
    n = len(idxs)
    corr = float(np.corrcoef(a_true, a_hat)[0, 1])
    global_med = np.median(a_hat)
    is_high = a_hat >= global_med

    # actual proportional x positions based on real frame timestamps
    x = _frame_x_positions(frames)   # [n], range [0, n-1] but non-uniform spacing

    n_thumb = min(8, n)
    thumb_pos = np.round(np.linspace(0, n - 1, n_thumb)).astype(int)

    fig = plt.figure(figsize=(n_thumb * 2.3, 9))
    gs_outer = gridspec.GridSpec(3, 1, figure=fig,
                                 height_ratios=[2.5, 0.10, 2.2],
                                 hspace=0.08)

    # ── Row 0: thumbnails ─────────────────────────────────────────────────
    gs_thumb = gridspec.GridSpecFromSubplotSpec(
        1, n_thumb, subplot_spec=gs_outer[0], wspace=0.04)

    for ti, fi in enumerate(thumb_pos):
        ax = fig.add_subplot(gs_thumb[0, ti])
        fname = os.path.basename(im_files[idxs[fi]])
        img_path = Path(image_dir) / fname
        if img_path.exists():
            img = Image.open(img_path).convert("RGB").resize((320, 180))
            ax.imshow(np.array(img))
        else:
            ax.set_facecolor("#555555")
        color = "#e63946" if is_high[fi] else "#2196F3"
        label = "HIGH" if is_high[fi] else "LOW"
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"t={fi+1}  {label}\nÂ={a_hat[fi]:.3f}",
                      color=color, fontsize=8, fontweight="bold")

    # ── Row 1: Â bar (uniform cells — thumbnail strip) ─────────────────────
    ax_bar = fig.add_subplot(gs_outer[1])
    norm_hat = (a_hat - a_hat.min()) / (a_hat.ptp() + 1e-9)
    ax_bar.imshow(norm_hat.reshape(1, -1), cmap="RdYlGn",
                  vmin=0, vmax=1, aspect="auto")
    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    ax_bar.set_ylabel("Â", fontsize=8, rotation=0, labelpad=20, va="center")
    for fi in thumb_pos:
        ax_bar.axvline(fi, color="white", linewidth=1.5, alpha=0.7)

    # ── Row 2: dual-axis line plot (x = real proportional frame positions) ──
    ax_line = fig.add_subplot(gs_outer[2])

    color_true = "crimson"
    color_hat  = "steelblue"

    l1, = ax_line.plot(x, a_true, color=color_true, linewidth=2.2,
                       marker="o", markersize=5, label="True A (left axis)")
    ax_line.fill_between(x, 0, a_true, alpha=0.12, color=color_true)
    ax_line.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_line.set_ylabel("True Advantage A", color=color_true, fontsize=10)
    ax_line.tick_params(axis="y", labelcolor=color_true)
    ax_line.set_xlim(x[0], x[-1])
    ax_line.grid(alpha=0.3)
    ax_line.set_xlabel("Sampled frame position (proportional to actual timestamp gap)", fontsize=10)

    ax2 = ax_line.twinx()
    l2, = ax2.plot(x, a_hat, color=color_hat, linewidth=2.0,
                   linestyle="--", marker="s", markersize=5,
                   label="Predicted Â (right axis)")
    ax2.set_ylabel("Predicted Â", color=color_hat, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color_hat)

    # match y-axis zero crossing visually
    at_min, at_max = a_true.min(), a_true.max()
    ap_min, ap_max = a_hat.min(), a_hat.max()
    at_pad = (at_max - at_min) * 0.15
    ap_pad = (ap_max - ap_min) * 0.15
    ax_line.set_ylim(at_min - at_pad, at_max + at_pad)
    ax2.set_ylim(ap_min - ap_pad, ap_max + ap_pad)

    lines = [l1, l2]
    labels_leg = [l.get_label() for l in lines]
    ax_line.legend(lines, labels_leg, fontsize=9, loc="upper left")

    # mark sampled frame positions on the line plot
    for fi in thumb_pos:
        ax_line.axvline(x[fi], color="gray", linewidth=0.8, alpha=0.5, linestyle=":")

    fig.suptitle(
        f"Temporal Routing Confidence — Clip '{vid}'  ({n} frames, thumbnails sampled)\n"
        f"True A ∈ [{a_true.min():.3f}, {a_true.max():.3f}]  |  "
        f"Predicted Â ∈ [{a_hat.min():.3f}, {a_hat.max():.3f}]",
        fontsize=12, fontweight="bold"
    )

    out_path = Path(out_dir) / f"11_temporal_tracking_clip{clip_i+1}_{vid}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache",  default="method_advantage_regress/outputs/bdd100k/cache_val_both.pt")
    p.add_argument("--router", default="method_advantage_regress/outputs/bdd100k/router_both_0.pt")
    p.add_argument("--images", default="/media/data/bdd100k_yolo/val/images")
    p.add_argument("--out",    default="method_advantage_regress/outputs/figures/router_spatial_temporal")
    p.add_argument("--clips",  nargs="*", default=None)
    p.add_argument("--n",      type=int, default=3)
    args = p.parse_args()

    print("Loading cache and router...")
    cache    = torch.load(args.cache, map_location="cpu")
    net      = load_router(args.router)
    adv_true = (cache["loss_base"] - cache["loss_super"]).numpy()
    adv_pred = predict_advantage(net, cache["input_base"], cache["pred_base"])
    im_files = cache["im_file"]

    if args.clips:
        video_frames = defaultdict(list)
        for idx, f in enumerate(im_files):
            vid = os.path.basename(f).split("-")[0]
            video_frames[vid].append((os.path.basename(f), idx))
        for vid in video_frames:
            video_frames[vid].sort(key=lambda x: x[0])

        clips = []
        for vid in args.clips:
            if vid not in video_frames:
                print(f"  [warn] '{vid}' not in cache")
                continue
            frames = video_frames[vid]
            idxs = [f[1] for f in frames]
            at = adv_true[idxs]
            ap = adv_pred[idxs]
            corr = float(np.corrcoef(at, ap)[0, 1]) if at.std() > 1e-6 else 0.0
            clips.append((vid, frames, corr, float(at.max() - at.min())))
    else:
        clips = auto_select_clips(adv_true, adv_pred, im_files, n=args.n)
        print("Auto-selected clips:")
        for vid, _, corr, ar in clips:
            print(f"  {vid}  r={corr:.3f}  A_range={ar:.3f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for i, (vid, frames, corr, _) in enumerate(clips):
        idxs = [f[1] for f in frames]
        plot_clip(vid, frames, idxs,
                  adv_true[idxs], adv_pred[idxs],
                  im_files, args.images, args.out, i)

    print("Done.")


if __name__ == "__main__":
    main()
