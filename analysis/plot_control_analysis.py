"""PI controller analysis figure: step response, parameter sensitivity, disturbance rejection.

Visualization signal: centered MA(win=30) — matches the tracking figures in the paper.
Metric computation: causal MA(20) — avoids non-causal smoothing artefacts.

Panels:
  (a) Step-up response  — time-aligned at the budget step-up, 3 families overlaid
  (b) Step-down response — same for step-down
  (c) Parameter sensitivity — beta sensitivity (dominant) + gain robustness
  (d) Disturbance rejection — worst-case condition change (night->day) within constant target

Usage:
    python -m step4_deploy.plot_control_analysis \
        --out results/step4_deploy/control/control_analysis.pdf
"""
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "bdd100k"

JSONS_BETA = {
    r"$\beta{=}0.83$": OUTPUTS / "jetson_trtengine_720x1280_b0.83_kp2.0_ki0.33_win45.json",
    r"$\beta{=}0.85$": OUTPUTS / "jetson_trtengine_720x1280_b0.85_kp2.0_ki0.33.json",
    r"$\beta{=}0.93$": OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp2.0_ki0.33.json",
}
JSONS_GAIN = {
    r"$k_p{=}2,k_i{=}0.33$": OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp2.0_ki0.33.json",
    r"$k_p{=}2,k_i{=}2.0$":  OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp2.0_ki2.0.json",
    r"$k_p{=}4,k_i{=}0.80$": OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp4.0_ki0.80.json",
}
JSON_OPT = OUTPUTS / "jetson_trtengine_720x1280_b0.85_kp2.0_ki0.33.json"
BETA = 0.85


# ─── signal helpers ──────────────────────────────────────────────────────────

def centered_ma(x, n):
    """Centered (non-causal) moving average — used for display."""
    pad = n // 2
    xp  = np.pad(x, (pad, n - pad - 1), mode='edge')
    return np.convolve(xp, np.ones(n) / n, mode='valid')


def causal_ma(x, n):
    """Causal (past-only) MA — used for step-response metrics."""
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = x[max(0, i - n + 1):i + 1].mean()
    return out


def load_maes(path, win=30):
    d    = json.loads(Path(path).read_text())
    step, saw = [], []
    for k, cell in d["cells"].items():
        sm  = centered_ma(np.asarray(cell["realized"]), win)
        mae = np.mean(np.abs(sm - np.asarray(cell["target"])))
        (step if "step" in k else saw).append(mae)
    return np.mean(step), np.mean(saw)


# ─── step response ────────────────────────────────────────────────────────────

def extract_step_windows(d, win_disp=30, win_metric=20, pre=60, post=180):
    """
    Return aligned windows for step-up and step-down transitions.
    Each entry: dict with 'sm' (display), 'cma' (metric), 'tgt_lo', 'tgt_hi'.
    """
    up, dn = [], []
    for fam in d['fam_order']:
        cell  = d['cells'][f'{fam}/step']
        raw   = np.asarray(cell['realized'], dtype=float)
        tgt   = np.asarray(cell['target'],   dtype=float)
        sm    = centered_ma(raw, win_disp)
        cma   = causal_ma(raw, win_metric)
        diffs = np.diff(tgt)
        for t_idx in np.where(np.abs(diffs) > 0.5)[0]:
            t0 = t_idx + 1
            if t0 < pre or t0 + post > len(raw):
                continue
            delta = tgt[t0] - tgt[t0 - 1]
            sl    = slice(t0 - pre, t0 + post)
            entry = dict(sm=sm[sl], cma=cma[sl],
                         tgt_lo=min(tgt[t0], tgt[t0-1]),
                         tgt_hi=max(tgt[t0], tgt[t0-1]))
            (up if delta > 0 else dn).append(entry)
    return up, dn


def step_metrics_cma(entries, direction, pre, settle_band=1.5):
    """Average rise time, overshoot, settling time across entries using CMA signal."""
    rts, oss, sts = [], [], []
    for e in entries:
        cma      = e['cma']
        t_before = e['tgt_lo'] if direction == 'UP' else e['tgt_hi']
        t_after  = e['tgt_hi'] if direction == 'UP' else e['tgt_lo']
        delta    = t_after - t_before   # signed: +6.53 for UP, -3.92 for DN
        post     = cma[pre:]

        # 90% rise threshold in the direction of the step
        rise_thr = t_before + 0.9 * delta
        rt = next((i for i, v in enumerate(post)
                   if (delta > 0 and v >= rise_thr) or (delta < 0 and v <= rise_thr)), None)
        if rt is not None:
            rts.append(rt)

        # Overshoot: exceedance beyond t_after in the step direction
        if delta > 0:
            os_ms = max(0.0, post.max() - t_after)
        else:
            os_ms = max(0.0, t_after - post.min())
        oss.append(os_ms / abs(delta) * 100)

        # Settling: stay within ±settle_band of t_after for 10 consecutive frames
        st = next((i for i in range(len(post) - 10)
                   if np.all(np.abs(post[i:i + 10] - t_after) <= settle_band)), None)
        if st is not None:
            sts.append(st)

    rt_avg = int(np.round(np.mean(rts))) if rts else None
    os_avg = float(np.mean(oss))
    st_avg = int(np.round(np.mean(sts))) if sts else None
    return rt_avg, os_avg, st_avg


# ─── disturbance rejection ────────────────────────────────────────────────────

def find_best_disturbance(d, win_disp=30):
    """Return the condition-boundary event with the largest peak SM deviation."""
    HALF = 60     # show 60 frames before and after the condition change
    POST = 80     # search window for peak/recovery after the boundary
    best = None
    for fam in d['fam_order']:
        cell   = d['cells'][f'{fam}/step']
        raw    = np.asarray(cell['realized'], dtype=float)
        tgt    = np.asarray(cell['target'],   dtype=float)
        sm     = centered_ma(raw, win_disp)
        bounds = d['families'][fam]['bounds']
        labels = d['families'][fam]['labels']
        diffs  = np.diff(tgt)
        tgt_trans = set(np.where(np.abs(diffs) > 0.5)[0] + 1)

        for k in range(1, len(bounds) - 1):
            b = bounds[k]
            if b < HALF or b + max(POST, HALF) > len(raw):
                continue
            if any(t in range(b - 30, b + 30) for t in tgt_trans):
                continue
            tgt_val    = float(tgt[b])
            post_sm    = sm[b:b + POST]
            post_tgt   = tgt[b:b + POST]
            deviations = np.abs(post_sm - post_tgt)
            peak_dev   = float(deviations.max())
            peak_idx   = int(deviations.argmax())   # relative to b
            # Recovery: first frame within ±1ms for 5 consecutive frames
            recovery = next((i for i in range(len(post_sm) - 5)
                             if np.all(np.abs(post_sm[i:i+5] - post_tgt[i:i+5]) <= 1.0)), None)
            if best is None or peak_dev > best['peak_dev']:
                # Clip peak_idx to the display window if needed
                disp_peak = min(peak_idx, HALF - 1)
                best = dict(
                    fam=fam, bound=b,
                    label_before=labels[k - 1], label_after=labels[k],
                    tgt=tgt_val, peak_dev=peak_dev,
                    peak_idx=disp_peak,       # relative to b, clipped to HALF
                    recovery=min(recovery, HALF - 1) if recovery is not None else None,
                    sm=sm[b - HALF:b + HALF],
                    tgt_arr=tgt[b - HALF:b + HALF],
                    pre_mae=float(np.mean(np.abs(sm[b - HALF:b] - tgt_val))),
                    HALF=HALF,
                )
    return best


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUTS / "control_analysis.pdf"))
    ap.add_argument("--win",    type=int, default=30)
    ap.add_argument("--cma",    type=int, default=20)
    ap.add_argument("--pre",    type=int, default=60)
    ap.add_argument("--settle", type=float, default=1.5)
    args = ap.parse_args()

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    d_opt = json.loads(Path(JSON_OPT).read_text())

    # ── Data ──────────────────────────────────────────────────────────────────
    up_entries, dn_entries = extract_step_windows(
        d_opt, win_disp=args.win, win_metric=args.cma,
        pre=args.pre, post=180)

    rt_up, os_up, st_up = step_metrics_cma(up_entries, 'UP', args.pre, args.settle)
    rt_dn, os_dn, st_dn = step_metrics_cma(dn_entries, 'DN', args.pre, args.settle)

    beta_data = {k: load_maes(v, args.win) for k, v in JSONS_BETA.items()}
    gain_data = {k: load_maes(v, args.win) for k, v in JSONS_GAIN.items()}

    evt = find_best_disturbance(d_opt, args.win)

    # ── Colors ────────────────────────────────────────────────────────────────
    FAM_COLORS = ["#5B9BD5", "#ED7D31", "#70AD47"]   # per-family traces
    C_TGT   = "black"
    C_BAND  = "#27AE60"
    C_RISE  = "#8E44AD"
    C_SETTL = "#E67E22"
    C_STEP  = "#4878CF"
    C_SAW   = "#D65F5F"
    C_DIST  = "#C0392B"

    PRE = args.pre
    xs  = np.arange(PRE + 180) - PRE

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Row 0: step-up | step-down | param sensitivity
    # Row 1: disturbance rejection (full width)
    fig = plt.figure(figsize=(7.2, 5.6))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0],
                           hspace=0.52, wspace=0.40,
                           left=0.09, right=0.97, top=0.95, bottom=0.10)

    ax_up   = fig.add_subplot(gs[0, 0])
    ax_dn   = fig.add_subplot(gs[0, 1], sharey=ax_up)
    ax_sen  = fig.add_subplot(gs[0, 2])
    ax_dist = fig.add_subplot(gs[1, :])

    # ── (a) Step-up  /  (b) Step-down ────────────────────────────────────────
    for ax, entries, direction, panel_label, rt, os_pct, st in [
        (ax_up,  up_entries, 'UP',   "(a) Step-up response",  rt_up, os_up, st_up),
        (ax_dn,  dn_entries, 'DOWN', "(b) Step-down response", rt_dn, os_dn, st_dn),
    ]:
        t_lo = float(np.mean([e['tgt_lo'] for e in entries]))
        t_hi = float(np.mean([e['tgt_hi'] for e in entries]))
        t_before = t_lo if direction == 'UP' else t_hi
        t_after  = t_hi if direction == 'UP' else t_lo
        delta    = t_after - t_before
        band     = args.settle

        # Individual family traces (SM, lightly)
        for i, e in enumerate(entries):
            ax.plot(xs, e['sm'], color=FAM_COLORS[i % 3], lw=1.1, alpha=0.45, zorder=2)

        # Mean SM across families (renormalize each to [t_before, t_after])
        norm_sms = [(e['sm'] - e['tgt_lo']) / (e['tgt_hi'] - e['tgt_lo'])
                    for e in entries]
        mean_norm = np.mean(norm_sms, axis=0)
        mean_sm   = mean_norm * (t_hi - t_lo) + t_lo
        ax.plot(xs, mean_sm, color="black", lw=1.7, zorder=5,
                label="mean (3 scenarios)")

        # Reference lines
        ax.axhline(t_before, color=C_TGT, ls=":", lw=0.9, alpha=0.5)
        ax.axhline(t_after,  color=C_TGT, ls="--", lw=1.1, zorder=6,
                   label=r"$B^\star(t)$")
        ax.axvline(0, color="0.55", ls=":", lw=0.9)

        # Settling band
        ax.axhspan(t_after - band, t_after + band,
                   color=C_BAND, alpha=0.13, zorder=1, label=f"±{band} ms settle")

        # Annotations — rise time bracket
        if rt is not None:
            rise_y = t_before + 0.9 * delta
            ax.annotate("", xy=(rt, rise_y), xytext=(0, rise_y),
                        arrowprops=dict(arrowstyle="<->", color=C_RISE, lw=1.1))
            ax.text(rt / 2, rise_y - 0.3 * abs(delta) * 0.12,
                    f"$T_r$={rt}f",
                    ha="center", va="top", fontsize=6.0, color=C_RISE)

        # Settling time bracket
        if st is not None:
            brkt_y = t_after + 0.55 * band * np.sign(delta)
            ax.annotate("", xy=(st, brkt_y), xytext=(0, brkt_y),
                        arrowprops=dict(arrowstyle="<->", color=C_SETTL, lw=1.1))
            ax.text(st / 2, brkt_y + 0.25 * abs(delta) * 0.12,
                    f"$T_s$={st}f",
                    ha="center", va="bottom", fontsize=6.0, color=C_SETTL)

        ax.set_xlim(-PRE, 180); ax.tick_params(labelsize=6.5)
        ax.set_xlabel("frame (rel. to step)", fontsize=7)
        ax.grid(alpha=0.18, ls="--")
        ax.set_title(panel_label, fontsize=8, pad=3)

        os_str = f"{os_pct:.0f}%"
        st_str = f"{st}f" if st is not None else "—"
        rt_str = f"{rt}f" if rt is not None else "—"
        ax.text(0.98, 0.04,
                f"$T_r$={rt_str}  $T_s$={st_str}  OS={os_str}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="0.30",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))

    ax_up.set_ylabel("latency (ms)", fontsize=7)
    plt.setp(ax_dn.get_yticklabels(), visible=False)

    # Legend for step panels
    fam_handles = [
        Line2D([0], [0], color=c, lw=1.1, alpha=0.6, label=lbl)
        for c, lbl in zip(FAM_COLORS,
                          ["night↔day", "city↔hwy", "clear↔rain"])
    ]
    fam_handles += [
        Line2D([0], [0], color="black",  lw=1.6,          label="mean"),
        Line2D([0], [0], color=C_TGT,    ls="--", lw=1.1, label=r"$B^\star$"),
        mpatches.Patch(color=C_BAND, alpha=0.4,            label=f"±{args.settle}ms"),
    ]
    ax_up.legend(handles=fam_handles, fontsize=5.2, frameon=True,
                 framealpha=0.88, loc="upper left", ncol=2)

    # ── (c) Parameter sensitivity ─────────────────────────────────────────────
    W = 0.19
    beta_xs = np.arange(len(beta_data)) * 0.52
    gain_xs = beta_xs[-1] + 0.72 + np.arange(len(gain_data)) * 0.52

    for i, (lbl, (s, w)) in enumerate(beta_data.items()):
        chosen = "0.85" in lbl
        lw = 1.8 if chosen else 0.6
        ax_sen.bar(beta_xs[i] - W/2, s, W, color=C_STEP, linewidth=lw,
                   edgecolor="black" if chosen else C_STEP)
        ax_sen.bar(beta_xs[i] + W/2, w, W, color=C_SAW,  linewidth=lw,
                   edgecolor="black" if chosen else C_SAW)
        if chosen:
            ax_sen.text(beta_xs[i], -0.10, "★", ha="center", va="top",
                        fontsize=8, color="black",
                        transform=ax_sen.get_xaxis_transform())

    for i, (lbl, (s, w)) in enumerate(gain_data.items()):
        ax_sen.bar(gain_xs[i] - W/2, s, W, color=C_STEP, alpha=0.50,
                   linewidth=0.6, edgecolor=C_STEP)
        ax_sen.bar(gain_xs[i] + W/2, w, W, color=C_SAW,  alpha=0.50,
                   linewidth=0.6, edgecolor=C_SAW)

    all_xs   = np.concatenate([beta_xs, gain_xs])
    all_labs = list(beta_data.keys()) + list(gain_data.keys())
    ax_sen.set_xticks(all_xs)
    ax_sen.set_xticklabels(all_labs, fontsize=4.6, rotation=22, ha="right")
    ax_sen.set_ylabel("MAE (ms)", fontsize=7)
    ax_sen.set_ylim(0, 1.25); ax_sen.tick_params(labelsize=6.0)
    ax_sen.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax_sen.grid(axis="y", alpha=0.20, ls="--")
    # group divider
    ax_sen.axvline((beta_xs[-1] + gain_xs[0]) / 2, color="0.7", ls="--", lw=0.7)
    mid_b = float(np.mean(beta_xs))
    mid_g = float(np.mean(gain_xs))
    ax_sen.text(mid_b, -0.34, r"$\beta$ sensitivity", ha="center", fontsize=6.2,
                color="0.35", transform=ax_sen.get_xaxis_transform())
    ax_sen.text(mid_g, -0.34, r"Gain robustness ($\beta{=}0.93$)", ha="center",
                fontsize=6.2, color="0.35", transform=ax_sen.get_xaxis_transform())
    sen_handles = [
        mpatches.Patch(color=C_STEP, label="Step budget"),
        mpatches.Patch(color=C_SAW,  label="Sawtooth budget"),
    ]
    ax_sen.legend(handles=sen_handles, fontsize=5.5, loc="upper left",
                  frameon=True, framealpha=0.88)
    ax_sen.set_title("(c) Parameter sensitivity", fontsize=8, pad=3)

    # ── (d) Disturbance rejection ─────────────────────────────────────────────
    HALF   = evt['HALF']
    xs_d   = np.arange(len(evt['sm'])) - HALF
    sm_d   = evt['sm']
    tgt_d  = evt['tgt_arr']
    tgt_v  = evt['tgt']

    ax_dist.axvspan(0, min(30, HALF),
                    color="#F39C12", alpha=0.10, zorder=0)
    ax_dist.axvline(0, color="#E67E22", ls="-", lw=1.2, zorder=3,
                    label=f"condition change: {evt['label_before']}→{evt['label_after']}")

    ax_dist.plot(xs_d, sm_d, color=C_DIST, lw=1.6, zorder=5,
                 label="realized (smoothed)")
    ax_dist.axhline(tgt_v, color=C_TGT, ls="--", lw=1.1, zorder=6,
                    label=r"$B^\star(t)$ = " + f"{tgt_v:.1f} ms")

    # ±1ms recovery band
    ax_dist.axhspan(tgt_v - 1.0, tgt_v + 1.0,
                    color=C_BAND, alpha=0.10, zorder=1, label="±1 ms recovery band")

    # Peak deviation arrow
    pid = evt['peak_idx']
    pv  = sm_d[HALF + pid]
    ax_dist.annotate("", xy=(pid, pv), xytext=(pid, tgt_v),
                     arrowprops=dict(arrowstyle="<->", color=C_RISE, lw=1.2))
    ax_dist.text(pid + 2, (pv + tgt_v) / 2,
                 f"peak dev.\n{evt['peak_dev']:.1f} ms", fontsize=6.2,
                 color=C_RISE, va="center")

    # Recovery marker
    if evt['recovery'] is not None:
        ax_dist.axvline(evt['recovery'], color=C_BAND, ls="--", lw=1.0, alpha=0.8)
        ax_dist.text(evt['recovery'] + 1, tgt_v + 0.6,
                     f"$T_{{rec}}$={evt['recovery']}f",
                     fontsize=6.2, color="#27AE60", va="bottom")

    # Steady-state band shading (pre)
    pre_lo = tgt_v - evt['pre_mae'] * 1.5; pre_hi = tgt_v + evt['pre_mae'] * 1.5
    ax_dist.axhspan(pre_lo, pre_hi, xmin=0.0, xmax=HALF / len(xs_d),
                    color="0.75", alpha=0.15, zorder=0)

    ax_dist.set_xlim(xs_d[0], xs_d[-1])
    ax_dist.set_xlabel("frame (rel. to condition change)", fontsize=7)
    ax_dist.set_ylabel("latency (ms)", fontsize=7)
    ax_dist.tick_params(labelsize=6.5)
    ax_dist.grid(alpha=0.18, ls="--")
    ax_dist.set_title(
        f"(d) Disturbance rejection — worst-case condition change "
        f"({evt['label_before']}→{evt['label_after']}, target constant at {tgt_v:.1f} ms)",
        fontsize=7.5, pad=3)
    ax_dist.legend(fontsize=5.8, ncol=2, frameon=True, framealpha=0.88,
                   loc="upper right")

    # ── Save & report ─────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    print(f"[*] -> {out}")

    print(f"\n── Step Response (causal MA={args.cma}f, settle band=±{args.settle}ms) ──")
    print(f"  Step-UP  : T_r={rt_up}f  OS={os_up:.0f}%  T_s={st_up}f")
    print(f"  Step-DOWN: T_r={rt_dn}f  OS={os_dn:.0f}%  T_s={st_dn}f")

    print(f"\n── Disturbance Rejection ──")
    print(f"  Event    : {evt['label_before']}→{evt['label_after']} @ frame {evt['bound']}")
    print(f"  Steady MAE before: {evt['pre_mae']:.2f} ms")
    print(f"  Peak deviation   : {evt['peak_dev']:.2f} ms")
    print(f"  Recovery (±1ms)  : {evt['recovery']}f")

    print(f"\n── Parameter Sensitivity (centered MA={args.win}f) ──")
    for lbl, (s, w) in {**beta_data, **gain_data}.items():
        star = " ←chosen" if "0.85" in lbl and "k_p" not in lbl else ""
        print(f"  {lbl:30s} step={s:.3f} ms  saw={w:.3f} ms{star}")


if __name__ == "__main__":
    main()
