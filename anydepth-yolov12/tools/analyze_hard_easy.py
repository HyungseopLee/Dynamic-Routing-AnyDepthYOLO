"""
Hard / Easy image analysis on BDD100K (Step 2).

For each model (Super-net, Base-net), split val images into three buckets by loss:
    easy = bottom q,  mid = middle 1-2q,  hard = top q     (default q = 0.25)

Then compare buckets on:
  2.1) attribute distribution        (weather / scene / timeofday)
  2.2) per-image object-size mix     (small / medium / large, COCO definition)
       per-image class distribution  (10 BDD classes)
       per-image #GT objects         (boxplot)

Outputs (into --outdir):
  - hard_easy_attrs.png    2 rows (Super, Base) x 3 cols (weather, scene, timeofday)
  - hard_easy_objects.png  2 rows (Super, Base) x 3 cols (size, class, n_gt)

Usage:
    python tools/analyze_hard_easy.py \
        --csv ./analysis/bdd100k-AnyDepth/per_image_loss.csv \
        --attr /media/data/bdd100k_yolo/val/attributes.json \
        --labels /media/data/bdd100k_yolo/val/labels \
        --imgsz 1280 720 \
        --quantile 0.25 \
        --outdir ./analysis/bdd100k-AnyDepth/easy-mid-hard
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WEATHER = {0: "clear", 1: "rainy", 2: "snowy", 3: "overcast", 4: "foggy", 5: "partly cloudy", -1: "undefined"}
SCENE   = {0: "city street", 1: "highway", 2: "residential", 3: "parking lot", 4: "tunnel", 5: "gas stations", -1: "undefined"}
TIME    = {0: "daytime", 1: "night", 2: "dawn/dusk", -1: "undefined"}
ATTR_MAPS = {"weather": WEATHER, "scene": SCENE, "timeofday": TIME}
CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike",
               "motor", "traffic light", "traffic sign", "train"]
SIZE_BINS = ["small", "medium", "large"]   # COCO: <32^2, [32^2,96^2), >=96^2
BUCKETS = ["easy", "mid", "hard"]
BUCKET_COLORS = {"easy": "tab:green", "mid": "tab:gray", "hard": "tab:red"}


def assign_buckets(series, q):
    lo, hi = series.quantile([q, 1 - q])
    def f(v):
        if v <= lo: return "easy"
        if v >= hi: return "hard"
        return "mid"
    return series.apply(f), float(lo), float(hi)


def load_label_features(stems, labels_dir, img_w, img_h):
    """Parse YOLO labels -> per-image (size_counts[3], class_counts[10], n_gt)."""
    rows = []
    for stem in stems:
        f = labels_dir / f"{stem}.txt"
        size_c = np.zeros(3, dtype=int)
        cls_c = np.zeros(len(CLASS_NAMES), dtype=int)
        n_gt = 0
        if f.exists():
            for line in f.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                w_n, h_n = float(parts[3]), float(parts[4])
                area = (w_n * img_w) * (h_n * img_h)
                if area < 32 * 32:
                    size_c[0] += 1
                elif area < 96 * 96:
                    size_c[1] += 1
                else:
                    size_c[2] += 1
                if 0 <= c < len(CLASS_NAMES):
                    cls_c[c] += 1
                n_gt += 1
        rows.append((stem, *size_c, *cls_c, n_gt))
    cols = ["stem"] + [f"size_{s}" for s in SIZE_BINS] \
           + [f"cls_{i}" for i in range(len(CLASS_NAMES))] + ["n_gt_lbl"]
    return pd.DataFrame(rows, columns=cols)


SIZE_COLORS = ["#2ca02c", "#ff7f0e", "#9467bd"]   # green / orange / purple
CLS_COLORS = plt.get_cmap("tab10").colors


def _count_share_matrix(df, bucket_col, count_cols):
    mat = np.zeros((len(BUCKETS), len(count_cols)))
    totals = np.zeros(len(BUCKETS), dtype=int)
    for bi, b in enumerate(BUCKETS):
        sub = df[df[bucket_col] == b]
        totals[bi] = len(sub)
        sums = sub[count_cols].sum(axis=0).to_numpy().astype(float)
        s = sums.sum()
        mat[bi] = sums / s if s > 0 else 0.0
    return mat, totals


def dodge_stacked_bar(ax, df, count_cols, names, colors, title, ylabel):
    """Per-bucket Super(solid) vs Base(hatched) dodged stacked bars over count_cols."""
    mat_s, tot_s = _count_share_matrix(df, "bucket_super", count_cols)
    mat_b, tot_b = _count_share_matrix(df, "bucket_base",  count_cols)
    x = np.arange(len(BUCKETS))
    w = 0.36
    pos_s = x - w / 2 - 0.02
    pos_b = x + w / 2 + 0.02

    for mat, pos, hatch in [(mat_s, pos_s, None), (mat_b, pos_b, "//")]:
        bottoms = np.zeros(len(BUCKETS))
        for ci, name in enumerate(names):
            vals = mat[:, ci]
            ax.bar(pos, vals, width=w, bottom=bottoms, color=colors[ci % len(colors)],
                   edgecolor="white", lw=0.5, hatch=hatch,
                   label=name if hatch is None else None)
            for bi, v in enumerate(vals):
                if v > 0.05:
                    ax.text(pos[bi], bottoms[bi] + v / 2, f"{v*100:.0f}",
                            ha="center", va="center", fontsize=6)
            bottoms += vals

    ax.set_xticks(x); ax.set_xticklabels(BUCKETS, fontsize=10)
    for bi in range(len(BUCKETS)):
        ax.text(pos_s[bi], 1.02, f"S\nn={tot_s[bi]}", ha="center", va="bottom", fontsize=7)
        ax.text(pos_b[bi], 1.02, f"B\nn={tot_b[bi]}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    cat_legend = ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.add_artist(cat_legend)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="lightgray", edgecolor="white", label="Super (S)"),
                       Patch(facecolor="lightgray", edgecolor="white", hatch="//", label="Base (B)")],
              loc="upper left", bbox_to_anchor=(1.02, 0.0), fontsize=7)


def _attr_share_matrix(df, bucket_col, attr, cats):
    """[bucket, cat] share of images; also returns per-bucket totals."""
    mat = np.zeros((len(BUCKETS), len(cats)))
    totals = np.zeros(len(BUCKETS), dtype=int)
    for bi, b in enumerate(BUCKETS):
        sub = df[(df[bucket_col] == b) & (df[attr] != -1)]
        totals[bi] = len(sub)
        if len(sub) == 0:
            continue
        vc = sub[attr + "_name"].value_counts(normalize=True)
        for ci, c in enumerate(cats):
            mat[bi, ci] = float(vc.get(c, 0.0))
    return mat, totals


def attr_dodge_stacked(ax, df, attr, title):
    """Super vs Base dodged stacked bars per bucket (image-share over attribute cats)."""
    name_map = ATTR_MAPS[attr]
    cats = [name_map[k] for k in sorted(name_map) if k != -1]
    cmap = plt.get_cmap("tab20")
    mat_s, tot_s = _attr_share_matrix(df, "bucket_super", attr, cats)
    mat_b, tot_b = _attr_share_matrix(df, "bucket_base",  attr, cats)

    x = np.arange(len(BUCKETS))
    w = 0.36
    pos_s = x - w / 2 - 0.02
    pos_b = x + w / 2 + 0.02

    for mat, pos, hatch in [(mat_s, pos_s, None), (mat_b, pos_b, "//")]:
        bottoms = np.zeros(len(BUCKETS))
        for ci, c in enumerate(cats):
            vals = mat[:, ci]
            ax.bar(pos, vals, width=w, bottom=bottoms, color=cmap(ci % 20),
                   edgecolor="white", lw=0.5, hatch=hatch,
                   label=c if hatch is None else None)
            for bi, v in enumerate(vals):
                if v > 0.05:
                    ax.text(pos[bi], bottoms[bi] + v / 2, f"{v*100:.0f}",
                            ha="center", va="center", fontsize=6)
            bottoms += vals

    # bucket labels and per-model n
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS, fontsize=10)
    for bi in range(len(BUCKETS)):
        ax.text(pos_s[bi], 1.02, f"S\nn={tot_s[bi]}", ha="center", va="bottom",
                fontsize=7, color="black")
        ax.text(pos_b[bi], 1.02, f"B\nn={tot_b[bi]}", ha="center", va="bottom",
                fontsize=7, color="black")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("share of images")
    ax.set_title(title, fontsize=10)
    # category legend (only solid bars), plus a small Super/Base swatch legend
    cat_legend = ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                           fontsize=7, title=attr)
    ax.add_artist(cat_legend)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="lightgray", edgecolor="white", label="Super (S)"),
                       Patch(facecolor="lightgray", edgecolor="white", hatch="//", label="Base (B)")],
              loc="upper left", bbox_to_anchor=(1.02, 0.0), fontsize=7)


def ngt_box_dodge(ax, df, title):
    """Per-bucket Super (blue) vs Base (red) n_gt boxplots, side-by-side."""
    data_s = [df.loc[df["bucket_super"] == b, "n_gt_lbl"].to_numpy() for b in BUCKETS]
    data_b = [df.loc[df["bucket_base"]  == b, "n_gt_lbl"].to_numpy() for b in BUCKETS]
    x = np.arange(len(BUCKETS)); w = 0.36
    pos_s = x - w / 2 - 0.02
    pos_b = x + w / 2 + 0.02
    bp_s = ax.boxplot(data_s, positions=pos_s, widths=w, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black", lw=1.2))
    bp_b = ax.boxplot(data_b, positions=pos_b, widths=w, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch in bp_s["boxes"]:
        patch.set_facecolor("tab:blue"); patch.set_alpha(0.6); patch.set_edgecolor("black")
    for patch in bp_b["boxes"]:
        patch.set_facecolor("tab:red");  patch.set_alpha(0.6); patch.set_edgecolor("black")
    means_s = [float(np.mean(v)) for v in data_s]
    means_b = [float(np.mean(v)) for v in data_b]
    ax.scatter(pos_s, means_s, marker="D", s=24, color="white", edgecolor="black", zorder=5)
    ax.scatter(pos_b, means_b, marker="D", s=24, color="white", edgecolor="black", zorder=5)
    for xp, m in zip(pos_s, means_s):
        ax.text(xp, m, f" {m:.1f}", fontsize=7, color="tab:blue", va="center", ha="left")
    for xp, m in zip(pos_b, means_b):
        ax.text(xp, m, f" {m:.1f}", fontsize=7, color="tab:red", va="center", ha="left")
    ax.set_xticks(x); ax.set_xticklabels(BUCKETS, fontsize=10)
    ax.set_xlim(-0.6, len(BUCKETS) - 0.4)
    ax.set_ylabel("# GT objects per image")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="tab:blue", alpha=0.6, edgecolor="black", label="Super"),
                       Patch(facecolor="tab:red",  alpha=0.6, edgecolor="black", label="Base")],
              loc="upper left", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--attr", required=True)
    ap.add_argument("--labels", required=True, help="YOLO val/labels directory")
    ap.add_argument("--imgsz", nargs=2, type=int, default=[1280, 720],
                    help="image (W H) used to convert normalized boxes to pixels")
    ap.add_argument("--quantile", type=float, default=0.25,
                    help="bucket quantile: easy=bottom q, hard=top q")
    ap.add_argument("--outdir", default="runs/loss_analysis")
    args = ap.parse_args()

    q = args.quantile
    assert 0 < q < 0.5, "quantile must be in (0, 0.5)"
    q_tag = f"q{int(round(q * 100)):02d}"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    attrs = json.load(open(args.attr))
    for a in ATTR_MAPS:
        df[a] = df["stem"].map(lambda s, a=a: attrs.get(s, {}).get(a, -1))
        df[a + "_name"] = df[a].map(ATTR_MAPS[a])

    feats = load_label_features(df["stem"].tolist(), Path(args.labels),
                                args.imgsz[0], args.imgsz[1])
    df = df.merge(feats, on="stem", how="left")
    df[["size_" + s for s in SIZE_BINS]] = df[["size_" + s for s in SIZE_BINS]].fillna(0).astype(int)
    df[[f"cls_{i}" for i in range(len(CLASS_NAMES))]] = \
        df[[f"cls_{i}" for i in range(len(CLASS_NAMES))]].fillna(0).astype(int)
    df["n_gt_lbl"] = df["n_gt_lbl"].fillna(0).astype(int)

    df["bucket_super"], lo_s, hi_s = assign_buckets(df["loss_super"], q)
    df["bucket_base"],  lo_b, hi_b = assign_buckets(df["loss_base"],  q)
    n = len(df)
    print(f"[*] n={n}, q={q}")
    print(f"    Super-net thresholds: easy<={lo_s:.3f}, hard>={hi_s:.3f}")
    print(f"    Base-net  thresholds: easy<={lo_b:.3f}, hard>={hi_b:.3f}")
    for tag, col in [("Super", "bucket_super"), ("Base", "bucket_base")]:
        vc = df[col].value_counts().reindex(BUCKETS)
        print(f"    {tag} bucket sizes: " + ", ".join(f"{b}={int(vc[b])}" for b in BUCKETS))

    # ===== Figure 1: hard_easy_attrs.png — Super (solid) vs Base (hatched) dodged =====
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for ci, attr in enumerate(ATTR_MAPS):
        title = f"{attr}  (S: easy≤{lo_s:.2f}/hard≥{hi_s:.2f}, "\
                f"B: easy≤{lo_b:.2f}/hard≥{hi_b:.2f})"
        attr_dodge_stacked(axes[ci], df, attr, title)
    fig.suptitle(f"Hard vs Easy bucket — attribute mix  (q={q}, n={n})  | "
                 f"left bar = Super (solid), right bar = Base (hatched)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / f"hard_easy_attrs_{q_tag}.png", dpi=150)
    plt.close(fig)

    # ===== Figure 2: hard_easy_objects.png — Super (solid) vs Base (hatched) dodged =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    dodge_stacked_bar(axes[0], df,
                      [f"size_{s}" for s in SIZE_BINS], SIZE_BINS, SIZE_COLORS,
                      title="object size  (small<32², large≥96²)",
                      ylabel="share of all GT objects")
    dodge_stacked_bar(axes[1], df,
                      [f"cls_{i}" for i in range(len(CLASS_NAMES))], CLASS_NAMES, CLS_COLORS,
                      title="class mix",
                      ylabel="share of all GT objects")
    fig.suptitle(f"Hard vs Easy bucket — object stats  (q={q}, n={n})  | "
                 f"left bar = Super, right bar = Base",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / f"hard_easy_objects_{q_tag}.png", dpi=150)
    plt.close(fig)

    print(f"[*] figures written to {outdir}")


if __name__ == "__main__":
    main()
