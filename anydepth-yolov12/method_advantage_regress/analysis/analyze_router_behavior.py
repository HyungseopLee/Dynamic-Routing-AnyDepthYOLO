"""Router behavior analysis: what drives high predicted advantage?

Five analyses:
  1. Scene attribute correlation  (timeofday / weather / scene)
  2. Object density correlation   (GT box count vs advantage)
  3. Spatial feature heatmap      (which 2x2 cell drives predictions)
  4. Top-K image grid             (high vs low advantage examples)
  5. Router calibration           (predicted Â vs true A)

Usage:
    python -m method_advantage_regress.analysis.analyze_router_behavior \
        --cache method_advantage_regress/outputs/bdd100k/cache_val_both.pt \
        --router method_advantage_regress/outputs/bdd100k/router_both_0.pt \
        --labels /media/data/bdd100k_yolo/val/labels \
        --attrs /media/data/bdd100k_yolo/val/attributes.json \
        --images /media/data/bdd100k_yolo/val/images \
        --out method_advantage_regress/outputs/figures/router_behavior
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── label mappings ────────────────────────────────────────────────────────────
WEATHER = {0: "clear", 1: "rainy", 2: "snowy", 3: "overcast", 4: "foggy",
           5: "partly cloudy", -1: "unknown"}
TIMEOFDAY = {0: "daytime", 1: "night", 2: "dawn/dusk", -1: "unknown"}
SCENE = {0: "city street", 1: "highway", 2: "residential",
         3: "parking lot", 4: "tunnel", 5: "gas stations", -1: "unknown"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_router(router_path: str):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from method_advantage_regress.router.router_net import RouterNetwork, GapMlpNet
    ckpt = torch.load(router_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    is_gap = not any(k.endswith("weight") and v.dim() == 4 for k, v in sd.items())
    # infer group_dim and head_dim from checkpoint shapes
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


def predict_advantage(net, input_feats: torch.Tensor, pred_feats: torch.Tensor) -> np.ndarray:
    """input_feats/pred_feats: [N, C, G, G]. Returns predicted logit [N]."""
    from method_advantage_regress.router.router_net import GapMlpNet
    is_gap = isinstance(net, GapMlpNet)
    path_id = torch.zeros(input_feats.shape[0], dtype=torch.long)
    with torch.no_grad():
        if is_gap:
            inp = input_feats.mean(dim=(2, 3))
            prd = pred_feats.mean(dim=(2, 3))
            logits = net.logit(inp, prd, path_id)
        else:
            logits = net.logit(input_feats, pred_feats, path_id)
    return logits.squeeze(-1).numpy()


def stem(path: str) -> str:
    return Path(path).stem


# ── analysis 1: scene attribute correlation ───────────────────────────────────

def plot_attribute_correlation(adv_true: np.ndarray,
                               adv_pred: np.ndarray,
                               im_files: list,
                               attrs: dict,
                               out: Path):
    attr_defs = [
        ("timeofday", TIMEOFDAY, "Time of Day"),
        ("weather",   WEATHER,   "Weather"),
        ("scene",     SCENE,     "Scene"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, mapping, title) in zip(axes, attr_defs):
        groups_true, groups_pred = {}, {}
        for i, f in enumerate(im_files):
            vid = stem(f)
            attr = attrs.get(vid, {}).get(key, -1)
            label = mapping.get(attr, "unknown")
            groups_true.setdefault(label, []).append(adv_true[i])
            groups_pred.setdefault(label, []).append(adv_pred[i])

        # sort by true advantage median
        labels_sorted = sorted(groups_true, key=lambda k: np.median(groups_true[k]))
        counts = [len(groups_true[k]) for k in labels_sorted]
        x = np.arange(len(labels_sorted))

        # true advantage: box
        bp_t = ax.boxplot([groups_true[k] for k in labels_sorted],
                          positions=x - 0.2, widths=0.35, patch_artist=True,
                          medianprops=dict(color="crimson", linewidth=2),
                          whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                          flierprops=dict(marker='.', markersize=1, alpha=0.3))
        # predicted advantage: box
        bp_p = ax.boxplot([groups_pred[k] for k in labels_sorted],
                          positions=x + 0.2, widths=0.35, patch_artist=True,
                          medianprops=dict(color="navy", linewidth=2),
                          whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                          flierprops=dict(marker='.', markersize=1, alpha=0.3))
        for patch in bp_t["boxes"]:
            patch.set_facecolor("tomato"); patch.set_alpha(0.6)
        for patch in bp_p["boxes"]:
            patch.set_facecolor("steelblue"); patch.set_alpha(0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels_sorted, counts)],
                           fontsize=8, rotation=20, ha="right")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Predicted Advantage")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend([bp_t["boxes"][0], bp_p["boxes"][0]],
                  ["True A", "Predicted Â"], fontsize=8, loc="upper left")

    fig.suptitle("True vs Predicted Advantage by Scene Attribute (BDD100K val)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out.with_suffix("") / "1_attribute_correlation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[1] Saved: {out_path}")


# ── analysis 2: object density correlation ────────────────────────────────────

def plot_density_correlation(adv_true: np.ndarray,
                             adv_pred: np.ndarray,
                             im_files: list,
                             labels_dir: Path,
                             out: Path):
    n_boxes = []
    for f in im_files:
        lbl = labels_dir / (stem(f) + ".txt")
        if lbl.exists():
            with open(lbl) as fp:
                lines = [l.strip() for l in fp if l.strip()]
            n_boxes.append(len(lines))
        else:
            n_boxes.append(0)
    n_boxes = np.array(n_boxes)

    # bin by box count
    bins = [0, 5, 10, 20, 40, 10000]
    bin_labels = ["0–4", "5–9", "10–19", "20–39", "40+"]
    groups_true, groups_pred = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (n_boxes >= lo) & (n_boxes < hi)
        groups_true.append(adv_true[mask])
        groups_pred.append(adv_pred[mask])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # boxplot: true vs predicted side by side
    x = np.arange(len(bin_labels))
    bp_t = axes[0].boxplot(groups_true, positions=x - 0.2, widths=0.35,
                           patch_artist=True, medianprops=dict(color="crimson", linewidth=2),
                           flierprops=dict(marker='.', markersize=1, alpha=0.3))
    bp_p = axes[0].boxplot(groups_pred, positions=x + 0.2, widths=0.35,
                           patch_artist=True, medianprops=dict(color="navy", linewidth=2),
                           flierprops=dict(marker='.', markersize=1, alpha=0.3))
    for patch in bp_t["boxes"]:
        patch.set_facecolor("tomato"); patch.set_alpha(0.6)
    for patch in bp_p["boxes"]:
        patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{bl}\n(n={len(g)})" for bl, g in zip(bin_labels, groups_true)], fontsize=9)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("GT box count")
    axes[0].set_ylabel("Advantage")
    axes[0].set_title("True vs Predicted Advantage by Object Density", fontweight="bold")
    axes[0].legend([bp_t["boxes"][0], bp_p["boxes"][0]], ["True A", "Predicted Â"], fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # scatter: both running means
    sort_idx = np.argsort(n_boxes)
    x_sorted = n_boxes[sort_idx]
    window = 200
    for y_arr, color, label in [(adv_true, "crimson", "True A"),
                                 (adv_pred, "steelblue", "Predicted Â")]:
        y_sorted = y_arr[sort_idx]
        rm = np.convolve(y_sorted, np.ones(window) / window, mode="valid")
        xm = x_sorted[window // 2: window // 2 + len(rm)]
        axes[1].scatter(n_boxes, y_arr, alpha=0.08, s=3, c=color)
        axes[1].plot(xm, rm, color=color, linewidth=2, label=f"{label} (w={window})")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("GT box count")
    axes[1].set_ylabel("Advantage")
    axes[1].set_title("Running Mean vs Object Count", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(-1, min(100, n_boxes.max() + 1))
    axes[1].grid(alpha=0.3)

    fig.suptitle("True vs Predicted Advantage by Object Density (BDD100K val)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out.with_suffix("") / "2_density_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[2] Saved: {out_path}")


# ── analysis 3: spatial feature heatmap ───────────────────────────────────────

def plot_spatial_heatmap(input_feats: torch.Tensor,
                         pred_feats: torch.Tensor,
                         adv_true: np.ndarray,
                         net,
                         out: Path):
    """
    Measure how much each 2x2 spatial cell contributes to the final prediction
    by ablating (zeroing) each cell and measuring the drop in predicted advantage.
    """
    G = input_feats.shape[-1]  # grid size (2)
    N = input_feats.shape[0]
    subset = min(N, 2000)
    idx = np.random.choice(N, subset, replace=False)
    inp_sub  = input_feats[idx]
    pred_sub = pred_feats[idx]
    adv_sub  = adv_true[idx]

    def infer(inp, prd):
        return predict_advantage(net, inp, prd)

    baseline_pred = infer(inp_sub, pred_sub)

    importance = np.zeros((G, G))
    for r in range(G):
        for c in range(G):
            abl_inp = inp_sub.clone(); abl_inp[:, :, r, c] = 0.0
            abl_prd = pred_sub.clone(); abl_prd[:, :, r, c] = 0.0
            ablated_pred = infer(abl_inp, abl_prd)
            delta = np.abs(baseline_pred - ablated_pred)
            importance[r, c] = delta.mean()

    # also compute: high-advantage images average feature magnitude per cell
    high_mask = adv_sub > np.percentile(adv_sub, 75)
    low_mask = adv_sub < np.percentile(adv_sub, 25)
    feat_mag_high = inp_sub[high_mask].abs().mean(dim=(0, 1)).numpy()  # [G, G]
    feat_mag_low  = inp_sub[low_mask].abs().mean(dim=(0, 1)).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, data, title, cmap in zip(
        axes,
        [importance, feat_mag_high - feat_mag_low, feat_mag_high],
        ["Ablation Importance\n(mean |Δpred| per cell)",
         "Feature Magnitude Diff\n(high-adv − low-adv images)",
         "Feature Magnitude\n(top-25% advantage images)"],
        ["Reds", "RdBu_r", "Blues"]
    ):
        im = ax.imshow(data, cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(G)); ax.set_yticks(range(G))
        ax.set_xticklabels([f"col {i}" for i in range(G)])
        ax.set_yticklabels([f"row {i}" for i in range(G)])
        for r in range(G):
            for c in range(G):
                ax.text(c, r, f"{data[r, c]:.3f}", ha="center", va="center",
                        fontsize=10, color="black")

    fig.suptitle(f"Spatial Feature Heatmap — 2×{G} Grid (BDD100K val, n={subset})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out.with_suffix("") / "3_spatial_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] Saved: {out_path}")


# ── analysis 4: top-K image grid ──────────────────────────────────────────────

def plot_topk_images(adv_pred: np.ndarray,
                     adv_true: np.ndarray,
                     im_files: list,
                     images_dir: Path,
                     out: Path,
                     K: int = 12):
    fig, axes = plt.subplots(4, K, figsize=(K * 2.2, 4 * 2.5))

    configs = [
        ("High predicted (top)", np.argsort(adv_pred)[-K:][::-1], adv_pred, adv_true),
        ("Low predicted (bottom)", np.argsort(adv_pred)[:K],       adv_pred, adv_true),
        ("High true advantage",  np.argsort(adv_true)[-K:][::-1], adv_true, adv_pred),
        ("Low true advantage",   np.argsort(adv_true)[:K],         adv_true, adv_pred),
    ]

    for row, (row_title, idxs, primary, secondary) in enumerate(configs):
        axes[row, 0].set_ylabel(row_title, fontsize=9, fontweight="bold", rotation=90,
                                labelpad=4)
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            img_path = images_dir / (stem(im_files[i]) + ".jpg")
            if not img_path.exists():
                img_path = images_dir / Path(im_files[i]).name
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((220, 130), Image.BILINEAR)
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#ddd")
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Â={primary[i]:.2f}\nA={secondary[i]:.2f}",
                         fontsize=7, pad=2)

    fig.suptitle("Top-K Router Examples (BDD100K val)\n"
                 "Â = predicted advantage, A = true advantage",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out.with_suffix("") / "4_topk_images.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[4] Saved: {out_path}")


# ── analysis 5: router calibration ────────────────────────────────────────────

def plot_calibration(adv_true: np.ndarray,
                     adv_pred: np.ndarray,
                     out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- scatter: pred vs true ---
    ax = axes[0]
    lim = max(np.abs(adv_true).max(), np.abs(adv_pred).max()) * 1.05
    h = ax.hexbin(adv_true, adv_pred, gridsize=60, cmap="YlOrRd", mincnt=1)
    plt.colorbar(h, ax=ax, label="count")
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="perfect")
    ax.set_xlabel("True Advantage A"); ax.set_ylabel("Predicted Advantage Â")
    ax.set_title("Calibration: Â vs A", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- error vs true advantage bins ---
    ax = axes[1]
    bins = np.percentile(adv_true, np.linspace(0, 100, 11))
    bin_centers, mean_err, std_err = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (adv_true >= lo) & (adv_true < hi)
        err = adv_pred[mask] - adv_true[mask]
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            mean_err.append(err.mean())
            std_err.append(err.std())
    bin_centers = np.array(bin_centers)
    mean_err = np.array(mean_err)
    std_err = np.array(std_err)
    ax.fill_between(bin_centers, mean_err - std_err, mean_err + std_err,
                    alpha=0.3, color="steelblue")
    ax.plot(bin_centers, mean_err, color="steelblue", marker="o", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("True Advantage A (bin center)")
    ax.set_ylabel("Prediction Error  Â − A")
    ax.set_title("Prediction Bias by Advantage Bin", fontweight="bold")
    ax.grid(alpha=0.3)

    # --- histogram of errors ---
    ax = axes[2]
    err = adv_pred - adv_true
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    corr = float(np.corrcoef(adv_true, adv_pred)[0, 1])
    ax.hist(err, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="crimson", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Prediction Error  Â − A")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution", fontweight="bold")
    ax.text(0.97, 0.95, f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nCorr={corr:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.grid(alpha=0.3)

    fig.suptitle("Router Calibration Analysis (BDD100K val)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out.with_suffix("") / "5_calibration.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"[5] Saved: {out_path}")
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  Pearson r={corr:.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache",   default="method_advantage_regress/outputs/bdd100k/cache_val_both.pt")
    p.add_argument("--router",  default="method_advantage_regress/outputs/bdd100k/router_both_0.pt")
    p.add_argument("--labels",  default="/media/data/bdd100k_yolo/val/labels")
    p.add_argument("--attrs",   default="/media/data/bdd100k_yolo/val/attributes.json")
    p.add_argument("--images",  default="/media/data/bdd100k_yolo/val/images")
    p.add_argument("--out",     default="method_advantage_regress/outputs/figures/router_behavior")
    p.add_argument("--topk",    type=int, default=12)
    p.add_argument("--seed",    type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading cache...")
    cache = torch.load(args.cache, map_location="cpu")
    loss_base  = cache["loss_base"].numpy()   # [N]
    loss_super = cache["loss_super"].numpy()  # [N]
    adv_true   = loss_base - loss_super       # [N]
    im_files   = cache["im_file"]
    input_feats = cache["input_base"]         # [N, C, G, G]
    pred_feats  = cache["pred_base"]          # [N, C, G, G]

    print("Loading router network...")
    net = load_router(args.router)

    print("Running inference...")
    batch = 512
    preds = []
    for i in range(0, len(input_feats), batch):
        preds.append(predict_advantage(net, input_feats[i:i+batch], pred_feats[i:i+batch]))
    adv_pred = np.concatenate(preds)

    attrs = json.load(open(args.attrs))
    labels_dir = Path(args.labels)
    images_dir = Path(args.images)

    print("\n=== Analysis 1: Scene Attribute Correlation ===")
    plot_attribute_correlation(adv_true, adv_pred, im_files, attrs, out)

    print("\n=== Analysis 2: Object Density Correlation ===")
    plot_density_correlation(adv_true, adv_pred, im_files, labels_dir, out)

    print("\n=== Analysis 3: Spatial Feature Heatmap ===")
    plot_spatial_heatmap(input_feats, pred_feats, adv_true, net, out)

    print("\n=== Analysis 4: Top-K Image Grid ===")
    plot_topk_images(adv_pred, adv_true, im_files, images_dir, out, K=args.topk)


    print("\n=== Analysis 5: Router Calibration ===")
    plot_calibration(adv_true, adv_pred, out)

    print(f"\nAll figures saved to: {out}/")


if __name__ == "__main__":
    main()
