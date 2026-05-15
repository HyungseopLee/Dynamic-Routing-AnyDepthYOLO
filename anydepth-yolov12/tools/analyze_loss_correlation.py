"""
Analyze per-image detection loss vs BDD100K attributes, for Super-net and Base-net.

Inputs:
  - per_image_loss.csv  (from per_image_loss.py, with super/base columns)
  - attributes.json     (stem -> {weather, scene, timeofday})

Outputs (4 figures into --outdir):
  - loss_overview.png         : 2x2 panel
      [TL] super vs base joint scatter (+ y=x, fit, Pearson/Spearman)
      [TR] overlay histogram with q25/q75
      [BL] sorted per-image loss (super & base on same axis)
      [BR] bucket agreement 3x3 confusion + Jaccard(hard/easy)
  - bucket_attrs_super.png    : 1x3 stacked bars for {weather, scene, timeofday},
                                bucketed by Super-net loss (low25/mid/high25)
  - bucket_attrs_base.png     : same, bucketed by Base-net loss
  - mean_loss_by_attrs.png    : 1x3 side-by-side bars per attr class

Usage:
    python tools/analyze_loss_correlation.py \
        --csv runs/loss_analysis/per_image_loss.csv \
        --attr /media/data/bdd100k_yolo/val/attributes.json \
        --outdir runs/loss_analysis \
        2>&1 | tee ./runs/loss_analysis/analyze.log
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
MAPS = {"weather": WEATHER, "scene": SCENE, "timeofday": TIME}
BUCKETS = ["low25", "mid", "high25"]


def assign_buckets(series):
    q25, q75 = series.quantile([0.25, 0.75])
    def f(v):
        if v <= q25: return "low25"
        if v >= q75: return "high25"
        return "mid"
    return series.apply(f), float(q25), float(q75)


def stacked_bar(ax, ct, title, cmap, legend_title):
    ct = ct.loc[:, (ct.sum(axis=0) > 0)]
    totals = ct.sum(axis=1).astype(int)
    ratio = ct.div(totals.replace(0, 1), axis=0)
    bottoms = np.zeros(len(ct))
    for i, col in enumerate(ratio.columns):
        vals = ratio[col].to_numpy()
        counts = ct[col].to_numpy()
        ax.bar(ct.index, vals, bottom=bottoms, color=cmap(i % 20),
               label=col, edgecolor="white", linewidth=0.5)
        for j, (v, c, b) in enumerate(zip(vals, counts, bottoms)):
            if v > 0.03:
                ax.text(j, b + v / 2, f"{v*100:.0f}%\n({int(c)})",
                        ha="center", va="center", fontsize=7)
        bottoms += vals
    for j, nt in enumerate(totals):
        ax.text(j, 1.01, f"n={int(nt)}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("ratio")
    ax.set_title(title, fontsize=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title=legend_title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--attr", required=True)
    ap.add_argument("--outdir", default="runs/loss_analysis")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "loss_super" not in df.columns:
        raise SystemExit("CSV must contain loss_super/loss_base (rerun per_image_loss.py).")
    attrs = json.load(open(args.attr))
    for a in ("weather", "scene", "timeofday"):
        df[a] = df["stem"].map(lambda s, a=a: attrs.get(s, {}).get(a, -1))

    df["bucket_super"], q25s, q75s = assign_buckets(df["loss_super"])
    df["bucket_base"],  q25b, q75b = assign_buckets(df["loss_base"])
    n = len(df)

    pear, _ = pearsonr(df["loss_super"], df["loss_base"])
    spear, _ = spearmanr(df["loss_super"], df["loss_base"])
    super_high = set(df.index[df["bucket_super"] == "high25"])
    base_high  = set(df.index[df["bucket_base"]  == "high25"])
    super_low  = set(df.index[df["bucket_super"] == "low25"])
    base_low   = set(df.index[df["bucket_base"]  == "low25"])
    j_hard = len(super_high & base_high) / max(len(super_high | base_high), 1)
    j_easy = len(super_low  & base_low)  / max(len(super_low  | base_low),  1)
    print(f"[*] n={n}")
    print(f"[*] super q25={q25s:.3f}, q75={q75s:.3f}  | base q25={q25b:.3f}, q75={q75b:.3f}")
    print(f"[*] Pearson={pear:.4f}  Spearman={spear:.4f}")
    print(f"[*] Jaccard(hard high25)={j_hard:.3f}  Jaccard(easy low25)={j_easy:.3f}")

    # ===== 1) loss_overview.png =====
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # [TL] joint scatter
    ax = axes[0, 0]
    ax.scatter(df["loss_super"], df["loss_base"], s=4, alpha=0.35, color="tab:purple")
    lo = float(min(df["loss_super"].min(), df["loss_base"].min()))
    hi = float(max(df["loss_super"].max(), df["loss_base"].max()))
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1, label="y = x")
    a, b = np.polyfit(df["loss_super"], df["loss_base"], 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, a * xs + b, color="tab:red", lw=1.2, label=f"fit: y={a:.2f}x+{b:.2f}")
    ax.set_xlabel("Super-net loss"); ax.set_ylabel("Base-net loss")
    ax.set_title(f"Super vs Base (Pearson={pear:.3f}, Spearman={spear:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    # [TR] overlay histogram
    ax = axes[0, 1]
    bins = np.linspace(lo, hi, 80)
    ax.hist(df["loss_super"], bins=bins, color="tab:blue", alpha=0.5, label="Super")
    ax.hist(df["loss_base"],  bins=bins, color="tab:red",  alpha=0.5, label="Base")
    for q, c in [(q25s, "tab:blue"), (q75s, "tab:blue"), (q25b, "tab:red"), (q75b, "tab:red")]:
        ax.axvline(q, color=c, ls="--", lw=0.9)
    ax.set_xlabel("loss"); ax.set_ylabel("count")
    ax.set_title("Loss distribution (dashed = q25/q75)")
    ax.legend()

    # [BL] sorted per-image loss
    ax = axes[1, 0]
    order = df["loss_super"].sort_values().index
    x = np.arange(n)
    ax.scatter(x, df.loc[order, "loss_super"], s=3, alpha=0.5, color="tab:blue", label="Super")
    ax.scatter(x, df.loc[order, "loss_base"],  s=3, alpha=0.4, color="tab:red",  label="Base")
    ax.axhline(q25s, color="tab:blue", ls="--", lw=0.8); ax.axhline(q75s, color="tab:blue", ls="--", lw=0.8)
    ax.set_xlabel("image (sorted by Super loss)"); ax.set_ylabel("loss")
    ax.set_title("Per-image loss, sorted by Super")
    ax.legend()

    # [BR] bucket agreement confusion
    ax = axes[1, 1]
    conf = pd.crosstab(df["bucket_super"], df["bucket_base"]).reindex(
        index=BUCKETS, columns=BUCKETS).fillna(0).astype(int)
    im = ax.imshow(conf.values, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(BUCKETS)
    ax.set_yticks(range(3)); ax.set_yticklabels(BUCKETS)
    ax.set_xlabel("Base bucket"); ax.set_ylabel("Super bucket")
    ax.set_title(f"Bucket agreement | Jaccard(hard)={j_hard:.2f}, Jaccard(easy)={j_easy:.2f}")
    mx = conf.values.max()
    for i in range(3):
        for j in range(3):
            v = conf.values[i, j]
            ax.text(j, i, f"{v}\n({v/n*100:.1f}%)", ha="center", va="center",
                    color="white" if v > mx / 2 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Per-image loss overview (n={n})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "loss_overview.png", dpi=150)
    plt.close(fig)

    # ===== 2) bucket_attrs_{super,base}.png =====
    cmap = plt.get_cmap("tab20")
    for key, bcol, q25, q75 in [("super", "bucket_super", q25s, q75s),
                                 ("base",  "bucket_base",  q25b, q75b)]:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        for ax, (attr, name_map) in zip(axes, MAPS.items()):
            ct = pd.crosstab(df[bcol], df[attr]).reindex(BUCKETS).fillna(0).astype(int)
            ct = ct.rename(columns=name_map)
            stacked_bar(ax, ct, f"{attr}", cmap, legend_title=attr)
        fig.suptitle(f"[{key.title()}-net] attribute distribution per loss bucket "
                     f"(q25={q25:.3f}, q75={q75:.3f}, n={n})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(outdir / f"bucket_attrs_{key}.png", dpi=150)
        plt.close(fig)

    # ===== 3) mean_loss_by_attrs.png =====
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, (attr, name_map) in zip(axes, MAPS.items()):
        names_col = df[attr].map(name_map)
        grp_s = df.groupby(names_col)["loss_super"].agg(["mean", "count"])
        grp_b = df.groupby(names_col)["loss_base"].agg(["mean"])
        order = [name_map[k] for k in sorted(name_map) if name_map[k] in grp_s.index]
        grp_s = grp_s.loc[order]; grp_b = grp_b.loc[order]
        x = np.arange(len(order)); w = 0.38
        ax.bar(x - w/2, grp_s["mean"], w, color="tab:blue", label="Super", edgecolor="black", lw=0.4)
        ax.bar(x + w/2, grp_b["mean"], w, color="tab:red",  label="Base",  edgecolor="black", lw=0.4)
        for i, (ms, mb, c) in enumerate(zip(grp_s["mean"], grp_b["mean"], grp_s["count"])):
            ax.text(i - w/2, ms, f"{ms:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w/2, mb, f"{mb:.2f}\n(n={int(c)})", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
        ax.set_ylabel("mean loss"); ax.set_title(attr)
        ax.legend(loc="upper right", fontsize=8)
        ax.margins(y=0.20)
    fig.suptitle("Mean detection loss per attribute class", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "mean_loss_by_attrs.png", dpi=150)
    plt.close(fig)

    print(f"[*] 4 figures written to {outdir}")


if __name__ == "__main__":
    main()
