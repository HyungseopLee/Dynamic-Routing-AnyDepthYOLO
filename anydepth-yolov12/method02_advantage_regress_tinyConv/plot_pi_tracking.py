"""PI latency-budget control on BDD MOT-val: (a) tracking accuracy (realized vs
target SUPER-usage, PI vs fixed val-threshold) and (b) AP-vs-GFLOPs operating
points. The point: the PI loop hits any target latency online (no val cache),
while the val-derived fixed threshold over-shoots and scatters across seeds
(val->video transfer failure).

    python method02_advantage_regress_tinyConv/plot_pi_tracking.py \
        --curve outputs/bdd100k/eval/video_curve_pi.json \
        --out outputs/bdd100k/eval/fig_bdd_pi --metric map50
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PI_RE = re.compile(r"^policy_p\d+_pi(\d+)$")
BUD_RE = re.compile(r"^policy_p\d+_b(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="map50", choices=["map50", "map"])
    args = ap.parse_args()
    rows = json.loads(Path(args.curve).read_text())["rows"]

    pi, bud, anc = defaultdict(list), defaultdict(list), {}
    g_base = g_super = None
    for r in rows:
        if r["name"] == "always_base":  anc["base"] = r; g_base = r["gflops"]
        if r["name"] == "always_super": anc["super"] = r; g_super = r["gflops"]
        m = PI_RE.match(r["name"])
        if m: pi[int(m.group(1))].append(r); continue
        m = BUD_RE.match(r["name"])
        if m: bud[int(m.group(1))].append(r)

    def agg(d):  # target -> (target, mean_super%, sd_super%, mean_metric, mean_gflops)
        out = []
        for t in sorted(d):
            sr = np.array([x["super_rate"] for x in d[t]]) * 100
            mt = np.array([x[args.metric] for x in d[t]])
            gf = np.array([x["gflops"] for x in d[t]])
            out.append((t, sr.mean(), sr.std(), mt.mean(), mt.std(), gf.mean()))
        return np.array(out)

    P, Bd = agg(pi), agg(bud)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.4))

    # (a) tracking: realized vs target
    ax0.plot([0, 100], [0, 100], ":", color="gray", lw=1, label="ideal (realized = target)")
    ax0.errorbar(Bd[:, 0], Bd[:, 1], yerr=Bd[:, 2], fmt="s--", color="tab:orange",
                 capsize=3, ms=5, lw=1.6, label="fixed val-threshold")
    ax0.errorbar(P[:, 0], P[:, 1], yerr=P[:, 2], fmt="o-", color="tab:red",
                 capsize=3, ms=5, lw=1.8, label="PI controller (ours)")
    ax0.set_xlabel("target SUPER usage (%)  = target latency")
    ax0.set_ylabel("realized SUPER usage (%)")
    ax0.set_title("(a) latency-budget tracking\n(±1 std over 5 seeds)")
    ax0.grid(alpha=0.3); ax0.legend(fontsize=9, loc="upper left")
    ax0.set_xlim(0, 100); ax0.set_ylim(0, 100)

    # (b) AP vs GFLOPs operating points
    ax1.plot(Bd[:, 5], Bd[:, 3], "s--", color="tab:orange", ms=5, lw=1.6, label="fixed val-threshold")
    ax1.plot(P[:, 5], P[:, 3], "o-", color="tab:red", ms=5, lw=1.8, label="PI controller (ours)")
    for k, c, mk in (("base", "darkgreen", "*"), ("super", "black", "*")):
        if k in anc:
            ax1.scatter([anc[k]["gflops"]], [anc[k][args.metric]], marker=mk, s=240, color=c, zorder=6)
            ax1.annotate(k, (anc[k]["gflops"], anc[k][args.metric]),
                         textcoords="offset points", xytext=(6, 6))
    ax1.set_xlabel("GFLOPs (per frame)"); ax1.set_ylabel(args.metric.upper())
    ax1.set_title("(b) accuracy vs compute")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="lower right")

    fig.suptitle("BDD100K MOT-val (200 clips): PI latency-budget control of the routing threshold",
                 fontsize=12, y=1.0)
    out = f"{args.out}_{ {'map50':'ap50','map':'ap5095'}[args.metric] }.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
