"""
Replot a focused KITTI tracking trade-off (confidence routing vs always-super/always-base)
from an existing per_strategy_summary.json — no need to re-run eval_video_dynamic_kitti.py.

Compared strategies (per request):
  - random_p000           = always base-net
  - random_p100           = always super-net
  - rule_mean_conf_ge_10  (τ sweep)
  - rule_mean_conf_ge_50  (τ sweep)
  - rule_top10_mean_conf  (τ sweep)
  - rule_top30_mean_conf  (τ sweep)

Usage:
    python tools/plot_tradeoff_focus.py \
        --json ./runs/kitti/tracking/dynamic/20260520_143502/per_strategy_summary.json \
        --out  ./runs/kitti/tracking/dynamic/20260520_143502/trade_off_focus.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FOCUS_FEATS = [
    ("mean_conf_ge_10", "tab:blue",   "o"),
    ("mean_conf_ge_25", "tab:cyan",   "s"),
    # ("mean_conf_all", "tab:orange", "^"),
    ("top20_mean_conf", "mediumorchid", "D"),
    ("top30_mean_conf", "tab:purple", "D"),
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]


def discover_features(overall):
    """Return sorted list of rule feature names present in the summary."""
    feats = set()
    for name in overall:
        parsed = parse_rule(name)
        if parsed is not None:
            feats.add(parsed[0])
    return sorted(feats)


def build_feat_styles(feats):
    """Assign (color, marker) to each feature deterministically."""
    cmap = plt.get_cmap("tab10")
    return [(f, cmap(i % 10), MARKERS[i % len(MARKERS)]) for i, f in enumerate(feats)]


def parse_rule(name):
    """rule_{feature}_t{TT} -> (feature, tau_int) or None."""
    if not name.startswith("rule_"):
        return None
    body = name[len("rule_"):]
    if "_t" not in body:
        return None
    feat, tt = body.rsplit("_t", 1)
    try:
        return feat, int(tt)
    except ValueError:
        return None


def x_value(s, xkey):
    if xkey == "pct_super": return 100.0 - s["pct_base"]
    if xkey == "latency":   return s["mean_latency_ms"]
    raise ValueError(xkey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="per_strategy_summary.json path")
    ap.add_argument("--out", default=None, help="output png path (default: same dir as json)")
    ap.add_argument("--all", action="store_true",
                    help="plot every rule feature found in the summary "
                         "(otherwise plot the curated FOCUS_FEATS list)")
    args = ap.parse_args()

    json_path = Path(args.json)
    summary = json.loads(json_path.read_text())
    overall = summary.get("overall") or summary  # tolerate flat dicts

    feats_to_plot = (build_feat_styles(discover_features(overall))
                     if args.all else FOCUS_FEATS)
    print(f"[*] plotting {len(feats_to_plot)} feature(s): "
          + ", ".join(f for f, *_ in feats_to_plot))

    ap_super = overall.get("random_p100", {}).get("mAP@50")
    ap_base  = overall.get("random_p000", {}).get("mAP@50")

    fig, ax_p = plt.subplots(1, 1, figsize=(8.5, 6))
    for xkey, ax, xlabel in [("pct_super", ax_p, "% super-net used")]:
        if ap_super is not None:
            ax.axhline(ap_super, color="tab:blue", ls=":", lw=1.2, alpha=0.9,
                       label=f"always_super (mAP@50={ap_super:.3f})")
        if ap_base is not None:
            ax.axhline(ap_base, color="tab:red", ls=":", lw=1.2, alpha=0.9,
                       label=f"always_base (mAP@50={ap_base:.3f})")
        # Linear-interpolation baseline between always-base and always-super
        # (= expected random_pXX curve, since mAP averages linearly with P(super)).
        if ap_super is not None and ap_base is not None:
            ax.plot([0.0, 100.0], [ap_base, ap_super], color="gray", ls="-",
                    lw=1.0, alpha=0.6, zorder=2, label="base–super linear interp")
        # Random sweep (baseline trade-off curve from random_p000 .. random_p100).
        rand_pts = []
        for name, s in overall.items():
            if not name.startswith("random_p"): continue
            try:
                p = int(name[len("random_p"):])
            except ValueError:
                continue
            rand_pts.append((x_value(s, xkey), s["mAP@50"], p))
        if rand_pts:
            rand_pts.sort(key=lambda t: t[0])
            rx = [p[0] for p in rand_pts]; ry = [p[1] for p in rand_pts]
            ax.plot(rx, ry, "--s", color="black", lw=1.2, ms=4, alpha=0.85,
                    zorder=3, label="random sweep")
        for feat, color, marker in feats_to_plot:
            pts = []
            for name, s in overall.items():
                parsed = parse_rule(name)
                if parsed is None: continue
                f_, tt = parsed
                if f_ != feat: continue
                pts.append((x_value(s, xkey), s["mAP@50"], tt))
            if not pts:
                print(f"[!] no points for {feat} — was this strategy run?")
                continue
            pts.sort(key=lambda t: t[0])
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            tts = [p[2] for p in pts]
            ax.plot(xs, ys, "-", color=color, lw=1.6, marker=marker, ms=5,
                    alpha=0.9, label=f"{feat} (τ sweep)")
            if xkey == "pct_super":
                for xv, yv, tt in zip(xs, ys, tts):
                    ax.annotate(f"{tt/100:.2f}", (xv, yv), fontsize=6,
                                xytext=(3, 3), textcoords="offset points",
                                color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP@50 (accumulated)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax_p.set_xlim(-5, 105)
    ax_p.set_title("trade-off vs %super-net used", fontsize=11)
    fig.suptitle("KITTI Tracking — confidence-based routing vs always-super/always-base",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = Path(args.out) if args.out else json_path.with_name("trade_off_focus.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] figure -> {out}")


if __name__ == "__main__":
    main()
