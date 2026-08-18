"""
What predicts Δ_AP = ap_super − ap_base ?

For every scalar feature available in the CSV (plus object stats from YOLO
labels and attribute one-hots), compute Pearson/Spearman correlation with
Δ_AP. Rank by |Spearman|.

Outputs:
  - apdiff_feature_ranking.png   horizontal bar of top-25 features
  - apdiff_topk_scatter.png      scatter for top-6 features (with fit + r)
  - console: full correlation table

Usage:
    python tools/analyze_apdiff_features.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_loss_pr_conf.csv \
        --attr /media/data/bdd100k_yolo/val/attributes.json \
        --labels /media/data/bdd100k_yolo/val/labels \
        --imgsz 1280 720 \
        --outdir ./analysis/bdd100k-AnyDepth/apdiff-features
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


WEATHER = {0: "clear", 1: "rainy", 2: "snowy", 3: "overcast", 4: "foggy", 5: "partly cloudy", -1: "undefined"}
SCENE   = {0: "city street", 1: "highway", 2: "residential", 3: "parking lot", 4: "tunnel", 5: "gas stations", -1: "undefined"}
TIME    = {0: "daytime", 1: "night", 2: "dawn/dusk", -1: "undefined"}
ATTR_MAPS = {"weather": WEATHER, "scene": SCENE, "timeofday": TIME}
CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike",
               "motor", "traffic light", "traffic sign", "train"]
SIZE_BINS = ["small", "medium", "large"]


def load_label_features(stems, labels_dir, img_w, img_h):
    rows = []
    for stem in stems:
        f = labels_dir / f"{stem}.txt"
        size_c = np.zeros(3, dtype=int)
        cls_c = np.zeros(len(CLASS_NAMES), dtype=int)
        n_gt = 0
        if f.exists():
            for line in f.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5: continue
                c = int(parts[0])
                w_n, h_n = float(parts[3]), float(parts[4])
                area = (w_n * img_w) * (h_n * img_h)
                if area < 32 * 32: size_c[0] += 1
                elif area < 96 * 96: size_c[1] += 1
                else: size_c[2] += 1
                if 0 <= c < len(CLASS_NAMES): cls_c[c] += 1
                n_gt += 1
        rows.append((stem, *size_c, *cls_c, n_gt))
    cols = ["stem"] + [f"size_{s}" for s in SIZE_BINS] \
           + [f"cls_{i}" for i in range(len(CLASS_NAMES))] + ["n_gt_lbl"]
    return pd.DataFrame(rows, columns=cols)


def build_feature_dict(df):
    """Return dict[name] -> 1-D float array.  All scalar features to correlate with Δ_AP."""
    feats = {}

    # (A) Super vs Base prediction differences (over-detection probe)
    for c in ["mean_conf_10", "mean_conf_25", "mean_conf_50", "mean_conf_75",
              "mean_conf_all", "top10_mean_conf"]:
        feats[f"diff_{c}"] = df[f"{c}_super"].to_numpy(float) - df[f"{c}_base"].to_numpy(float)
    for c in ["n_pred_10", "n_pred_25", "n_pred_50", "n_pred_75", "n_pred_all"]:
        feats[f"diff_{c}"] = df[f"{c}_super"].to_numpy(float) - df[f"{c}_base"].to_numpy(float)
    feats["diff_tp"] = df["tp_super"].to_numpy(float) - df["tp_base"].to_numpy(float)
    feats["diff_fp"] = df["fp_super"].to_numpy(float) - df["fp_base"].to_numpy(float)
    feats["diff_precision"] = df["precision_super"].to_numpy(float) - df["precision_base"].to_numpy(float)
    feats["diff_recall"]    = df["recall_super"].to_numpy(float)    - df["recall_base"].to_numpy(float)

    # (B) Super-net stand-alone signals
    for c in ["loss_super", "box_super", "cls_super", "dfl_super",
              "precision_super", "recall_super",
              "mean_conf_25_super", "mean_conf_50_super", "top10_mean_conf_super",
              "n_pred_25_super", "n_pred_all_super"]:
        feats[c] = df[c].to_numpy(float)

    # (C) Base-net stand-alone signals (could be input to decision net)
    for c in ["loss_base", "precision_base", "recall_base",
              "mean_conf_25_base", "mean_conf_50_base", "top10_mean_conf_base",
              "n_pred_25_base", "n_pred_all_base"]:
        feats[c] = df[c].to_numpy(float)

    # (D) Object stats (per-image)
    feats["n_gt"] = df["n_gt_lbl"].to_numpy(float)
    n_gt_safe = np.maximum(df["n_gt_lbl"].to_numpy(float), 1)
    for s in SIZE_BINS:
        feats[f"size_{s}_share"] = df[f"size_{s}"].to_numpy(float) / n_gt_safe
        feats[f"size_{s}_count"] = df[f"size_{s}"].to_numpy(float)
    for i, name in enumerate(CLASS_NAMES):
        feats[f"cls_{name}_share"] = df[f"cls_{i}"].to_numpy(float) / n_gt_safe
        feats[f"cls_{name}_count"] = df[f"cls_{i}"].to_numpy(float)

    # (E) Attribute one-hots (drop undefined)
    for attr in ATTR_MAPS:
        for k, name in ATTR_MAPS[attr].items():
            if k == -1: continue
            feats[f"{attr}={name}"] = (df[attr] == k).astype(float).to_numpy()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--attr", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--imgsz", nargs=2, type=int, default=[1280, 720])
    ap.add_argument("--outdir", default="analysis/bdd100k-AnyDepth/apdiff-features")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    attrs = json.load(open(args.attr))
    for a in ATTR_MAPS:
        df[a] = df["stem"].map(lambda s, a=a: attrs.get(s, {}).get(a, -1))
    feats_obj = load_label_features(df["stem"].tolist(), Path(args.labels),
                                    args.imgsz[0], args.imgsz[1])
    df = df.merge(feats_obj, on="stem", how="left")
    df[["size_" + s for s in SIZE_BINS]] = df[["size_" + s for s in SIZE_BINS]].fillna(0).astype(int)
    df[[f"cls_{i}" for i in range(len(CLASS_NAMES))]] = \
        df[[f"cls_{i}" for i in range(len(CLASS_NAMES))]].fillna(0).astype(int)
    df["n_gt_lbl"] = df["n_gt_lbl"].fillna(0).astype(int)

    y = (df["ap5095_super"] - df["ap5095_base"]).to_numpy(float)
    y_mask = np.isfinite(y)
    print(f"[*] n={len(df)}, n_valid_Δ={int(y_mask.sum())}")
    print(f"    Δ_AP: mean={y[y_mask].mean():+.4f}, median={np.median(y[y_mask]):+.4f}, "
          f"std={y[y_mask].std():.4f}")
    print(f"    Δ>0 (Super better): {(y[y_mask]>0).sum()} "
          f"({(y[y_mask]>0).mean()*100:.1f}%)")
    print(f"    Δ<0 (Base  better): {(y[y_mask]<0).sum()} "
          f"({(y[y_mask]<0).mean()*100:.1f}%)")

    feats = build_feature_dict(df)
    rows = []
    for name, x in feats.items():
        m = y_mask & np.isfinite(x)
        if m.sum() < 100 or len(np.unique(x[m])) < 2:
            continue
        try:
            pear = pearsonr(x[m], y[m])[0]
            spear = spearmanr(x[m], y[m])[0]
        except Exception:
            continue
        rows.append((name, pear, spear, abs(spear)))
    rank = pd.DataFrame(rows, columns=["feature", "Pearson", "Spearman", "abs_S"]) \
             .sort_values("abs_S", ascending=False).reset_index(drop=True)

    print("\n[*] Full correlation table (sorted by |Spearman|):")
    print(rank.to_string(index=False, formatters={
        "Pearson":  "{:+.3f}".format, "Spearman": "{:+.3f}".format,
        "abs_S":    "{:.3f}".format}))

    # ===== Figure 1: top-25 ranking =====
    top = rank.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = ["tab:green" if s > 0 else "tab:red" for s in top["Spearman"]]
    ax.barh(top["feature"], top["Spearman"], color=colors, edgecolor="black", lw=0.4)
    ax.axvline(0, color="black", lw=1)
    for i, (s, p) in enumerate(zip(top["Spearman"], top["Pearson"])):
        ax.text(s, i, f"  S={s:+.3f}, P={p:+.3f}", va="center",
                ha="left" if s >= 0 else "right", fontsize=8)
    ax.set_xlabel("Spearman ρ with Δ_AP   (green: ρ>0 → larger feature ⇒ Super wins more,  "
                  "red: ρ<0 → larger feature ⇒ Base wins more)")
    ax.set_title(f"Feature ranking by |Spearman| correlation with Δ_AP "
                 f"(top-25 of {len(rank)}, n={int(y_mask.sum())})",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "apdiff_feature_ranking.png", dpi=150)
    plt.close(fig)

    # ===== Figure 2: top-6 scatter =====
    top6 = rank.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (_, row) in zip(axes.flat, top6.iterrows()):
        name = row["feature"]
        x = feats[name]
        m = y_mask & np.isfinite(x)
        ax.scatter(x[m], y[m], s=3, alpha=0.25, color="tab:purple", edgecolors="none")
        if x[m].std() > 1e-9:
            a, b = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(float(x[m].min()), float(x[m].max()), 100)
            ax.plot(xs, a * xs + b, color="black", ls="--", lw=1.2,
                    label=f"y={a:.3f}x{b:+.3f}")
            ax.legend(loc="best", fontsize=8)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xlabel(name); ax.set_ylabel("Δ_AP = ap_super − ap_base")
        ax.set_title(f"{name}\nPearson={row['Pearson']:+.3f}  Spearman={row['Spearman']:+.3f}",
                     fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle("Top-6 features by |Spearman| with Δ_AP",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "apdiff_topk_scatter.png", dpi=150)
    plt.close(fig)

    print(f"\n[*] figures -> {outdir}")


if __name__ == "__main__":
    main()
