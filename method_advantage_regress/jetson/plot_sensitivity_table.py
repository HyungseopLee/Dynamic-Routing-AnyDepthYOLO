"""Hyperparameter sensitivity table — one-at-a-time sweep format.

3 groups, each fixing two parameters and varying the third:
  Group 1: vary α    (k_p=2.0, k_i=0.33 fixed)
  Group 2: vary k_p  (k_i=0.33, α=0.85 fixed)
  Group 3: vary k_i  (k_p=2.0,  α=0.85 fixed)

Columns match the paper table format:
  α | k_p | k_i | Rise↑/↓ | Settling↑/↓ | OS↑/↓ | SSE Step/Saw

Usage:
    python -m method_advantage_regress.jetson.plot_sensitivity_table \
        --out method_advantage_regress/outputs/bdd100k/sensitivity_table.pdf
"""
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "bdd100k"

# ── table row definitions ─────────────────────────────────────────────────────
# Each entry: (alpha, kp, ki, json_path, is_chosen)  or  None = group separator
ROWS = [
    # ── Group 1: vary α  (k_p=2.0, k_i=0.33 fixed) ──────────────────────────
    ("0.83", "2.0", "0.33",
     OUTPUTS / "jetson_trtengine_720x1280_b0.83_kp2.0_ki0.33_win45.json",
     False),
    ("0.85", "2.0", "0.33",
     OUTPUTS / "jetson_trtengine_720x1280_b0.85_kp2.0_ki0.33.json",
     True),
    ("0.93", "2.0", "0.33",
     OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp2.0_ki0.33.json",
     False),
    None,
    # ── Group 2: vary k_p  (k_i=0.33, α=0.85 fixed) ─────────────────────────
    ("0.85", "0.5", "0.33",
     OUTPUTS / "jetson_trtengine_720×1280_b0.85_kp0.5_ki0.33_warmup60_window30.json",
     False),
    ("0.85", "5.0", "0.33",
     OUTPUTS / "jetson_trtengine_720×1280_b0.85_kp5.0_ki0.33_warmup60_window30.json",
     False),
    None,
    # ── Group 3: vary k_i  (k_p=2.0, α=0.85 fixed) ──────────────────────────
    ("0.85", "2.0", "0.10",
     OUTPUTS / "jetson_trtengine_720×1280_b0.85_kp2.0_ki0.1_warmup60_window30.json",
     False),
    ("0.85", "2.0", "1.00",
     OUTPUTS / "jetson_trtengine_720×1280_b0.85_kp2.0_ki1.0_warmup60_window30.json",
     False),
]

# ── signal helpers ────────────────────────────────────────────────────────────

def centered_ma(x, n):
    pad = n // 2
    xp  = np.pad(x, (pad, n - pad - 1), mode='edge')
    return np.convolve(xp, np.ones(n) / n, mode='valid')


def causal_ma(x, n):
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = x[max(0, i - n + 1):i + 1].mean()
    return out


# ── metrics ───────────────────────────────────────────────────────────────────

CHATTER_THRESH = 0.80   # transition rate above this → chattering regime


def switching_rate(realized, l_base, l_super):
    mid = (l_base + l_super) / 2
    choices = (np.asarray(realized) > mid).astype(int)
    return np.mean(np.diff(choices) != 0)


def compute_metrics(path, cma_win=20, disp_win=30, pre=60, settle_band=1.5,
                    use_fps=False):
    p = Path(path)
    if not p.exists():
        return None

    d = json.loads(p.read_text())
    sse_step, sse_saw = [], []
    up_tr, up_ts, up_os = [], [], []
    dn_tr, dn_ts, dn_os = [], [], []
    chatter_rates = []

    # infer l_base / l_super from realized latency distribution
    all_realized = np.concatenate([
        np.asarray(d['cells'][f'{fam}/step']['realized'])
        for fam in d['fam_order']
    ])
    l_base  = float(np.percentile(all_realized, 10))
    l_super = float(np.percentile(all_realized, 90))

    for fam in d['fam_order']:
        for btype in ('step', 'sawtooth'):
            cell = d['cells'][f'{fam}/{btype}']
            raw  = np.asarray(cell['realized'], dtype=float)
            tgt  = np.asarray(cell['target'],   dtype=float)

            # SSE: optionally in FPS (convert before smoothing to preserve nonlinearity)
            if use_fps:
                sm  = centered_ma(1000.0 / raw, disp_win)
                mae = float(np.mean(np.abs(sm - 1000.0 / tgt)))
            else:
                sm  = centered_ma(raw, disp_win)
                mae = float(np.mean(np.abs(sm - tgt)))
            (sse_step if btype == 'step' else sse_saw).append(mae)

            if btype != 'step':
                continue

            # Rise / Settling / OS: always in latency space (controller dynamics)
            chatter_rates.append(switching_rate(raw, l_base, l_super))
            cma   = causal_ma(raw, cma_win)
            diffs = np.diff(tgt)
            for t_idx in np.where(np.abs(diffs) > 0.5)[0]:
                t0 = t_idx + 1
                if t0 < pre or t0 + 180 > len(raw):
                    continue
                delta    = tgt[t0] - tgt[t0 - 1]
                t_before = tgt[t0 - 1]
                t_after  = tgt[t0]
                post     = cma[t0:t0 + 180]

                rise_thr = t_before + 0.9 * delta
                tr = next((i for i, v in enumerate(post)
                           if (delta > 0 and v >= rise_thr) or
                              (delta < 0 and v <= rise_thr)), None)

                os_ms  = (max(0.0, post.max() - t_after) if delta > 0
                          else max(0.0, t_after - post.min()))
                os_pct = os_ms / abs(delta) * 100.0

                ts = next((i for i in range(len(post) - 10)
                           if np.all(np.abs(post[i:i + 10] - t_after) <= settle_band)),
                          None)

                if delta > 0:
                    if tr is not None: up_tr.append(tr)
                    up_os.append(os_pct)
                    if ts is not None: up_ts.append(ts)
                else:
                    if tr is not None: dn_tr.append(tr)
                    dn_os.append(os_pct)
                    if ts is not None: dn_ts.append(ts)

    def m(lst):
        return int(np.round(np.mean(lst))) if lst else None

    mean_chatter = float(np.mean(chatter_rates)) if chatter_rates else 0.0
    is_chattering = mean_chatter > CHATTER_THRESH

    return dict(
        sse_step     = float(np.mean(sse_step)),
        sse_saw      = float(np.mean(sse_saw)),
        tr_up        = m(up_tr),  ts_up = m(up_ts),
        os_up        = float(np.mean(up_os)) if up_os else None,
        tr_dn        = m(dn_tr),  ts_dn = m(dn_ts),
        os_dn        = float(np.mean(dn_os)) if dn_os else None,
        chatter_rate = mean_chatter,
        chattering   = is_chattering,
    )


# ── LaTeX output ─────────────────────────────────────────────────────────────

def fmt(v, dec=0, missing="---"):
    if v is None:
        return missing
    return f"{v:.{dec}f}" if dec else str(int(round(v)))


def fmt_os(val, chattering, latex=False):
    """Format an OS value; replace with chattering marker when applicable."""
    if chattering:
        return r"chat.$^\dagger$" if latex else "chat."
    if val is None:
        return "---"
    return f"{val:.1f}"


def print_latex(all_metrics, unit="ms"):
    print("\n% ── LaTeX ──────────────────────────────────────────────────────")
    print(r"""\begin{table}[t]
\caption{
    Sensitivity analysis of the PI controller parameters on BDD100K MOT
    (720$\times$1280, Jetson Orin Nano).
    Rise, Settling, OS, and SSE denote rise time, settling time, overshoot,
    and steady-state tracking error, respectively.
    Metrics are reported for step-up ($\uparrow$) and step-down ($\downarrow$)
    budget transitions.
    The row marked $^*$ is the chosen default.
    $^\dagger$Chattering regime (transition rate $>$60\%): OS is undefined
    because the smoothed signal never reaches the target level.
}
\label{tab:controller_sensitivity}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}ccccccc@{}}
\toprule
\multicolumn{3}{c}{Parameters} & \multicolumn{4}{c}{Control Metrics} \\
\cmidrule(r){1-3} \cmidrule(l){4-7}
$\alpha$ & $k_p$ & $k_i$
  & Rise (f)                      & Settling (f)                  & OS (\%)                       & SSE (""" + unit + r""")       \\
  &       &
  & ($\uparrow$/$\downarrow$)
  & ($\uparrow$/$\downarrow$)
  & ($\uparrow$/$\downarrow$)
  & (Step/Saw)        \\
\midrule""")

    group = 0
    first = True
    for row in ROWS:
        if row is None:
            print(r"\midrule")
            group += 1
            continue

        alpha, kp, ki, path, is_chosen = row
        m = all_metrics.get(str(path))

        if m:
            ch   = m['chattering']
            tr   = f"{fmt(m['tr_up'])} / {fmt(m['tr_dn'])}"
            ts   = f"{fmt(m['ts_up'])} / {fmt(m['ts_dn'])}"
            os_  = f"{fmt_os(m['os_up'], ch, latex=True)} / {fmt_os(m['os_dn'], ch, latex=True)}"
            sse  = f"{fmt(m['sse_step'],2)} / {fmt(m['sse_saw'],2)}"
        else:
            tr = ts = os_ = sse = "---"

        star = r"$^{*}$" if is_chosen else ""
        a    = f"{alpha}{star}"
        cells = [a, kp, ki, tr, ts, os_, sse]
        if is_chosen:
            cells = [r"\textbf{" + c + "}" for c in cells]
        print(" & ".join(cells) + r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


# ── figure ───────────────────────────────────────────────────────────────────

def make_table_figure(all_metrics, out_path):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # Collect data rows
    data_rows = []
    grp = 0
    for row in ROWS:
        if row is None:
            grp += 1
            continue
        alpha, kp, ki, path, is_chosen = row
        m = all_metrics.get(str(path))
        data_rows.append((alpha, kp, ki, m, is_chosen, grp))

    def fv(v, dec=0):
        if v is None: return "—"
        return f"{v:.{dec}f}" if dec else str(int(round(v)))

    def pair(a, b, dec=0):
        return f"{fv(a,dec)} / {fv(b,dec)}"

    col_headers = [
        r"$\alpha$", r"$k_p$", r"$k_i$",
        "Rise (f)\n↑ / ↓",
        "Settling (f)\n↑ / ↓",
        "OS (%)\n↑ / ↓",
        "SSE (ms)\nStep / Saw",
    ]
    n_cols = len(col_headers)

    fig = plt.figure(figsize=(6.8, 3.4))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ROW_H   = 0.090
    HDR_H   = 0.130
    x0, y0 = 0.04, 0.94
    col_ws  = [0.075, 0.065, 0.065, 0.145, 0.145, 0.145, 0.160]
    xs = [x0]
    for w in col_ws:
        xs.append(xs[-1] + w)

    # group header
    grp_h = 0.058
    for lbl, c0, c1, col in [("Parameters", 0, 2, "#4A6FA5"),
                               ("Control Metrics", 3, 6, "#5B7A5B")]:
        gx0, gx1 = xs[c0], xs[c1 + 1]
        ax.add_patch(plt.Rectangle(
            (gx0 + 0.002, y0 - grp_h), gx1 - gx0 - 0.004, grp_h,
            transform=ax.transAxes, facecolor=col, edgecolor="white", lw=0.7))
        ax.text((gx0 + gx1) / 2, y0 - grp_h / 2, lbl,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")

    # column headers
    hdr_y   = y0 - grp_h
    hdr_col = "#D5D5D5"
    for ci, hdr in enumerate(col_headers):
        cx = 0.5 * (xs[ci] + xs[ci + 1])
        ax.add_patch(plt.Rectangle(
            (xs[ci] + 0.001, hdr_y - HDR_H + 0.002),
            xs[ci + 1] - xs[ci] - 0.002, HDR_H - 0.004,
            transform=ax.transAxes, facecolor=hdr_col, edgecolor="white", lw=0.4))
        ax.text(cx, hdr_y - HDR_H / 2, hdr,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=5.8, color="#111111", fontweight="bold", linespacing=1.2)

    # numeric ranges for color coding (lower = better for all)
    def metric_vals(key, fn=lambda m: m):
        return [fn(r[3]) for r in data_rows if r[3] and fn(r[3]) is not None]

    def col_bg(val, vals):
        if not vals or val is None: return (1.0, 1.0, 1.0)
        lo, hi = min(vals), max(vals)
        if lo == hi: return (1.0, 1.0, 1.0)
        t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        if t <= 0.5:
            f = t * 2
            return (0.67 + f * 0.33, 0.86 + f * 0.14, 0.67 + f * 0.33)
        else:
            f = (t - 0.5) * 2
            return (1.0 - f * 0.05, 1.0 - f * 0.37, 1.0 - f * 0.37)

    # gather ranges per display column
    ranges = {
        3: metric_vals('tr',   lambda m: (m['tr_up'] or 0) + (m['tr_dn'] or 0)),
        4: metric_vals('ts',   lambda m: (m['ts_up'] or 0) + (m['ts_dn'] or 0)),
        5: metric_vals('os',   lambda m: (m['os_up'] or 0) + (m['os_dn'] or 0)),
        6: metric_vals('sse',  lambda m: m['sse_step'] + m['sse_saw']),
    }
    sort_vals = {
        3: lambda m: (m['tr_up'] or 0) + (m['tr_dn'] or 0),
        4: lambda m: (m['ts_up'] or 0) + (m['ts_dn'] or 0),
        5: lambda m: (m['os_up'] or 0) + (m['os_dn'] or 0),
        6: lambda m: m['sse_step'] + m['sse_saw'],
    }

    prev_grp = -1
    cur_y    = hdr_y - HDR_H

    for ri, (alpha, kp, ki, m, is_chosen, grp) in enumerate(data_rows):
        if grp != prev_grp and prev_grp >= 0:
            ax.plot([xs[0] + 0.001, xs[-1] - 0.001],
                    [cur_y + ROW_H * 0.05] * 2,
                    transform=ax.transAxes, color="#888", lw=0.7, ls="--")
        prev_grp = grp

        row_top = cur_y
        cur_y  -= ROW_H

        star = "*" if is_chosen else ""
        ch = m['chattering'] if m else False
        cells_txt = [
            f"{alpha}{star}", kp, ki,
            pair(m['tr_up'],  m['tr_dn'])                             if m else "—",
            pair(m['ts_up'],  m['ts_dn'])                             if m else "—",
            f"{fmt_os(m['os_up'],ch)} / {fmt_os(m['os_dn'],ch)}"     if m else "—",
            f"{fv(m['sse_step'],2)} / {fv(m['sse_saw'],2)}"          if m else "—",
        ]

        for ci, txt in enumerate(cells_txt):
            cx = 0.5 * (xs[ci] + xs[ci + 1])
            cy = row_top - ROW_H / 2

            if is_chosen:
                bg = "#FFF9C4"
            elif ci >= 3 and m:
                sv = sort_vals[ci](m)
                bg = col_bg(sv, ranges[ci])
            elif ci >= 3:
                bg = "#F0F0F0"
            else:
                bg = "#FAFAFA" if ri % 2 == 0 else (1.0, 1.0, 1.0)

            ax.add_patch(plt.Rectangle(
                (xs[ci] + 0.001, row_top - ROW_H + 0.002),
                xs[ci + 1] - xs[ci] - 0.002, ROW_H - 0.004,
                transform=ax.transAxes, facecolor=bg, edgecolor="white", lw=0.3))

            fw = "bold" if is_chosen else "normal"
            fc = "#B8860B" if is_chosen else ("#999" if not m and ci >= 3 else "#111")
            ax.text(cx, cy, txt, transform=ax.transAxes,
                    ha="center", va="center", fontsize=5.8, fontweight=fw, color=fc)

        if is_chosen:
            ax.text(xs[0] - 0.005, row_top - ROW_H / 2, "*",
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=8, color="#B8860B", fontweight="bold")

    total_h = grp_h + HDR_H + len(data_rows) * ROW_H
    ax.add_patch(plt.Rectangle(
        (xs[0], cur_y), xs[-1] - xs[0], total_h + 0.003,
        transform=ax.transAxes, facecolor="none", edgecolor="#999", lw=0.8))
    ax.plot([xs[3], xs[3]], [cur_y, y0],
            transform=ax.transAxes, color="#999", lw=0.6)

    ax.text(xs[0], cur_y - 0.04,
            r"* Chosen default. Rise/Settling: causal MA-20, ±1.5 ms band. "
            r"OS: causal MA-20. SSE: centered MA-30. "
            r"$^\dagger$chat. = chattering regime (transition rate >60%): OS undefined.",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=4.6, color="0.45")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    print(f"[*] -> {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",    default=str(OUTPUTS / "sensitivity_table.pdf"))
    ap.add_argument("--cma",    type=int,   default=20)
    ap.add_argument("--win",    type=int,   default=30)
    ap.add_argument("--settle", type=float, default=1.5,
                    help="settling band in the signal unit (ms or FPS)")
    ap.add_argument("--fps",    action="store_true",
                    help="convert latency→FPS; settle band is then in FPS")
    args = ap.parse_args()

    unit = "FPS" if args.fps else "ms"
    # default settle band: 1.5 ms  ≈  ~1.0 FPS near 30 FPS; allow override
    settle = args.settle

    all_metrics = {}
    print(f"\nMode: {'FPS' if args.fps else 'latency'}  settle_band=±{settle} {unit}")
    print(f"\n{'Config':<40} {'Tr↑/↓':>10} {'Ts↑/↓':>10} {'OS↑/↓':>12} {'SSE Stp/Saw':>16}")
    print("-" * 96)

    for row in ROWS:
        if row is None:
            print()
            continue
        alpha, kp, ki, path, is_chosen = row
        m = compute_metrics(path, args.cma, args.win, 60, settle, use_fps=args.fps)
        all_metrics[str(path)] = m
        star = "* " if is_chosen else "  "
        if m:
            ch = m['chattering']
            os_str = f"{'chat.':>5}/{'chat.':>5}" if ch else \
                     f"{fmt(m['os_up'],1):>5}/{fmt(m['os_dn'],1):>5}%"
            chatter_note = f"  [chatter {m['chatter_rate']*100:.0f}%]" if ch else ""
            print(f"{star}α={alpha} kp={kp} ki={ki:<5}  "
                  f"{fmt(m['tr_up'])}/{fmt(m['tr_dn']):>3}f  "
                  f"{fmt(m['ts_up'])}/{fmt(m['ts_dn']):>3}f  "
                  f"{os_str}  "
                  f"{fmt(m['sse_step'],3)}/{fmt(m['sse_saw'],3)} {unit}"
                  f"{chatter_note}")
        else:
            print(f"  α={alpha} kp={kp} ki={ki:<5}  (missing)")

    make_table_figure(all_metrics, args.out)
    print_latex(all_metrics, unit=unit)


if __name__ == "__main__":
    main()
