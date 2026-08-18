"""Target-latency-budget scheduling: closed-loop tau control over the video.

Replays the per-frame dump (dump_perframe_kitti.py) under a time-varying latency
target L*(t) and adjusts the router threshold tau every frame so the realized
per-frame latency tracks the budget. Two controllers are compared:

  PI            : EMA(realized latency) error -> tau <- tau - kp*e - ki*sum(e).
                  Lower tau routes more frames to SUPER (higher latency), so the
                  sign drives realized latency toward the target.
  FF-quantile   : feedforward. Convert L* to a target SUPER-rate
                  p* = (L* - l_base)/(l_super - l_base) and set tau to the
                  (1 - p*) quantile of recently observed A-hat (rolling window).

Per-frame cost is memoryless, so latency = l[choice] + l_router with the measured
RTX-3090 anchors; AP under the realized schedule is recomputed by pooling the kept
path's match records. Routing is causal/recursive: frame t's decision uses the
A-hat of the path executed on t-1 (reset at each sequence boundary).

    python -m analysis.sim_latency_budget \
        --dump results/step3_eval/kitti/perframe_dump.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import step3_eval.eval_utils as B

# measured RTX-3090 anchors (match Table tab:device KITTI column)
L_BASE, L_SUPER, L_ROUTER = 13.35, 20.87, 0.33


def target_schedule(kind, n):
    """Return L*(t) over n frames within the feasible band [~14, ~21] ms."""
    lo, hi = L_BASE + L_ROUTER + 0.6, L_SUPER + L_ROUTER - 0.6  # ~14.3 .. 20.6
    t = np.arange(n)
    if kind == "step":
        seg = np.array([0.30, 0.85, 0.15, 0.65, 0.45])  # fraction of band, per block
        edges = np.linspace(0, n, len(seg) + 1).astype(int)
        L = np.empty(n)
        for i, frac in enumerate(seg):
            L[edges[i]:edges[i + 1]] = lo + frac * (hi - lo)
        return L
    if kind == "sawtooth":
        period = max(1, n // 4)
        frac = (t % period) / period            # 0..1 ramp, repeating
        return lo + frac * (hi - lo)
    raise ValueError(kind)


def simulate(frames, Ltgt, controller, kp=0.020, ki=0.004, win=120,
             tau0=0.15, tau_lo=-0.15, tau_hi=0.60, beta=0.85, score=True):
    n = len(frames)
    tau = tau0
    integ = 0.0
    L_ema = 0.5 * (L_BASE + L_SUPER)
    prev_choice = None
    ahat_hist = []
    realized = np.empty(n)
    taus = np.empty(n)
    is_super = np.zeros(n)
    n_super = 0
    matches, gt_count = [], {}
    for i, fr in enumerate(frames):
        if fr["first"]:
            prev_choice = None
        # causal routing signal = A-hat of the path executed on the previous frame
        if prev_choice is None:
            choice = "base"  # first frame of a sequence defaults to base
        else:
            pv = fr["ahat_super"] if prev_choice == "super" else fr["ahat_base"]
            ahat_hist.append(pv)
            if controller == "ff":
                p_star = (Ltgt[i] - L_BASE) / (L_SUPER - L_BASE)
                p_star = float(np.clip(p_star, 0.0, 1.0))
                recent = ahat_hist[-win:]
                tau = float(np.quantile(recent, 1.0 - p_star)) if recent else tau0
            choice = "super" if pv > tau else "base"
        prev_choice = choice
        # realized latency with measured anchors
        lat = (L_SUPER if choice == "super" else L_BASE) + L_ROUTER
        realized[i] = lat
        taus[i] = tau
        is_super[i] = choice == "super"
        n_super += choice == "super"
        L_ema = beta * L_ema + (1 - beta) * lat
        # PI update for next frame (sign: realized<target -> e>0 -> lower tau -> more super)
        if controller == "pi":
            e = Ltgt[i] - L_ema
            integ += e
            tau = float(np.clip(tau - kp * e - ki * integ, tau_lo, tau_hi))
        # score the kept path (skip for fast tracking-only sweeps)
        if score:
            matches.extend(fr[f"match_{choice}"])
            for c, k in fr["gt_count"].items():
                gt_count[c] = gt_count.get(c, 0) + k
    map50, map5095 = (None, None)
    if score:
        _, map50, map5095 = B.dataset_map_multi_iou(matches, gt_count)
    return {"realized": realized, "tau": taus, "is_super": is_super,
            "super_rate": n_super / n, "map50": map50, "map": map5095}


def smooth(x, w=150, causal=False):
    """Moving average with edge padding.

    causal=False: centered window (offline analysis, symmetric, no lag).
    causal=True : trailing window over the previous w frames only (the online,
                  deployable readout, since future frames are unavailable at runtime).
    """
    x = np.asarray(x, dtype=float)
    if causal:
        xp = np.pad(x, (w - 1, 0), mode="edge")
        k = np.ones(w) / w
        return np.convolve(xp, k, mode="valid")
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="same")[pad:pad + len(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(Path(__file__).resolve().parent /
                    "outputs/kitti/perframe_dump.pkl"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent.parent /
                    "results/figures/fig_latency_budget.pdf"))
    args = ap.parse_args()
    frames = pickle.load(open(args.dump, "rb"))["frames"]
    n = len(frames)
    seq_starts = [i for i, fr in enumerate(frames) if fr["first"]]
    print(f"[*] {n} frames; feasible latency band "
          f"[{L_BASE+L_ROUTER:.2f}, {L_SUPER+L_ROUTER:.2f}] ms")
    print(f"[*] {len(seq_starts)} sequences; boundaries at {seq_starts}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    scenarios = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]

    # 1 row x 2 cols: realized per-frame latency tracking the time-varying target.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    print(f"\n{'scenario':<10}{'MAE(ms)':>9}{'SUPER%':>9}{'AP50':>8}{'AP':>8}")
    for col, (skind, stitle) in enumerate(scenarios):
        Ltgt = target_schedule(skind, n)
        res = simulate(frames, Ltgt, "pi")
        sm = smooth(res["realized"])
        mae = float(np.mean(np.abs(sm - Ltgt)))
        print(f"{skind:<10}{mae:>9.2f}{res['super_rate']*100:>8.1f}%"
              f"{res['map50']*100:>8.1f}{res['map']*100:>8.1f}")
        ax = axes[col]
        # video (sequence) boundaries: skip frame 0 to avoid a line on the y-axis
        for j, s in enumerate(seq_starts[1:]):
            ax.axvline(s, color="tab:blue", ls="-", lw=0.7, alpha=0.35, zorder=1,
                       label="video boundary" if j == 0 else None)
        ax.plot(Ltgt, color="black", ls="--", lw=1.5, label="target budget", zorder=6)
        ax.plot(sm, color="tab:red", lw=1.4, label="realized (ours)", zorder=5)
        ax.set_title(stitle, fontsize=10)
        ax.set_ylim(13.0, 21.8)
        ax.set_xlabel("frame", fontsize=9)
        ax.grid(alpha=0.25, ls="--"); ax.tick_params(labelsize=8)
        ax.text(0.03, 0.05, f"MAE = {mae:.2f} ms", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))
    axes[0].set_ylabel("latency (ms)", fontsize=9)
    axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout(pad=0.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] -> {args.out}")


if __name__ == "__main__":
    main()
