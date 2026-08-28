"""Realistic condition-drift budget tracking on BDD100K MOT.

Instead of abrupt KITTI sequence boundaries, we concatenate real MOT sequences in
an alternating scene-condition order (night<->dawn, city<->highway, clear<->rainy)
and show the PI controller holding a step / sawtooth latency budget as the scene
condition drifts. Each segment is one MOT sequence (~200 frames); shaded bands mark
the condition of each segment.

    python -m analysis.make_scenario_figures_bdd
"""
import json
import pickle
from pathlib import Path

import numpy as np

import analysis.sim_latency_budget as S

# BDD anchors (Table tab:device BDD column): base/super per-frame latency (ms).
S.L_BASE, S.L_SUPER, S.L_ROUTER = 24.76, 39.16, 0.40

OUT = Path(__file__).resolve().parent.parent / "results/step3_eval/bdd100k"
DUMP = OUT / "perframe_scenarios.pkl"
SCEN = OUT / "scenarios.json"
FIG = Path(__file__).resolve().parent.parent / "results/figures/fig_scenario_budget.pdf"

TITLES = {"night_dawn": "night $\\leftrightarrow$ daytime",
          "city_highway": "city $\\leftrightarrow$ highway",
          "clear_rainy": "clear $\\leftrightarrow$ rainy"}
SHORT = {"night": "night", "daytime": "day", "dawn/dusk": "dawn",
         "city street": "city", "highway": "hwy", "clear": "clear", "rainy": "rain"}


# Target band spans almost the full feasible [base, super] range, mirroring the
# KITTI figure's proportions (0.6 ms margin at each end of [base, super]).
MARGIN = 0.6  # ms


def target_band():
    base, sup = S.L_BASE + S.L_ROUTER, S.L_SUPER + S.L_ROUTER
    return base + MARGIN, sup - MARGIN


def target_schedule(kind, n):
    """L*(t) over a trackable sub-band of the feasible [base, super] latency range."""
    lo, hi = target_band()
    t = np.arange(n)
    if kind == "step":
        seg = np.array([0.30, 0.85, 0.15, 0.65, 0.45])
        edges = np.linspace(0, n, len(seg) + 1).astype(int)
        L = np.empty(n)
        for i, frac in enumerate(seg):
            L[edges[i]:edges[i + 1]] = lo + frac * (hi - lo)
        return L
    if kind == "sawtooth":
        period = max(1, n // 4)
        return lo + ((t % period) / period) * (hi - lo)
    raise ValueError(kind)


def build_stream(order, seqs):
    """Concatenate the chosen sequences; mark segment boundaries + conditions."""
    frames, bounds, labels = [], [], []
    for seg in order:
        fr = seqs[seg["seq"]]["frames"]
        bounds.append(len(frames))
        labels.append(seg["cond"])
        for j, f in enumerate(fr):
            g = dict(f); g["first"] = (j == 0)
            frames.append(g)
    bounds.append(len(frames))
    return frames, bounds, labels


def main():
    data = pickle.load(open(DUMP, "rb"))["seqs"]
    scen = json.loads(SCEN.read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    fams = list(scen.keys())
    budgets = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]
    fig, axes = plt.subplots(len(fams), 2, figsize=(7.2, 1.9 * len(fams)),
                             squeeze=False)
    tlo, thi = target_band()
    base, sup = S.L_BASE + S.L_ROUTER, S.L_SUPER + S.L_ROUTER
    ylo, yhi = base - 1.2, sup + 1.6          # show the full feasible band (KITTI-style)
    print(f"feasible band [{S.L_BASE+S.L_ROUTER:.1f}, {S.L_SUPER+S.L_ROUTER:.1f}] ms; "
          f"target band [{tlo:.1f}, {thi:.1f}] ms")
    print(f"\n{'scenario':<14}{'budget':<10}{'MAE(ms)':>9}{'SUPER%':>9}")

    for r, fam in enumerate(fams):
        frames, bounds, labels = build_stream(scen[fam], data)
        n = len(frames)
        for c, (bkind, btitle) in enumerate(budgets):
            Ltgt = target_schedule(bkind, n)
            res = S.simulate(frames, Ltgt, "pi", score=False)
            # A single frame executes exactly one depth, so the per-frame latency is
            # binary (base/super) and can never equal an intermediate budget. The
            # quantity the controller regulates is the *mean* latency (the SUPER
            # rate), read out as a trailing (causal) moving average over the previous
            # 30 frames -- the online readout available at deployment.
            sm = S.smooth(np.asarray(res["realized"]), 30, causal=True)
            mae = float(np.mean(np.abs(sm - Ltgt)))
            print(f"{fam:<14}{bkind:<10}{mae:>9.2f}{res['super_rate']*100:>8.1f}%")
            ax = axes[r][c]
            # shade each condition segment, alternating two tints; label it
            for k in range(len(labels)):
                x0, x1 = bounds[k], bounds[k + 1]
                tint = "tab:blue" if k % 2 == 0 else "tab:orange"
                ax.axvspan(x0, x1, color=tint, alpha=0.07, zorder=0)
                ax.text((x0 + x1) / 2, thi + 0.5, SHORT.get(labels[k], labels[k]),
                        ha="center", va="bottom", fontsize=6.5, color="0.3")
                if k > 0:
                    ax.axvline(x0, color="0.6", ls="-", lw=0.6, alpha=0.5, zorder=1)
            ax.plot(Ltgt, color="black", ls="--", lw=1.3, label="target budget", zorder=6)
            ax.plot(sm, color="tab:red", lw=1.5,
                    label="realized latency (moving average)", zorder=5)
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(0, n)
            ax.grid(alpha=0.2, ls="--"); ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(btitle, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{TITLES[fam]}\nlatency (ms)", fontsize=7.5)
            if r == len(fams) - 1:
                ax.set_xlabel("frame", fontsize=8)
            ax.text(0.02, 0.04, f"MAE={mae:.2f} ms", transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="black", ls="--", lw=1.3, label="target budget"),
               Line2D([0], [0], color="tab:red", lw=1.5,
                      label="realized latency (moving average)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.03, 1, 1))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] -> {FIG}")


if __name__ == "__main__":
    main()
