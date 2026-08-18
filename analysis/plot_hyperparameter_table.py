"""Generate hyperparameter sensitivity table (PDF figure + LaTeX source).

Columns: alpha, kp, ki | T_r↑ T_r↓ (rise) | T_s↑ T_s↓ (settle) |
         OS↑ OS↓ (overshoot) | SSE_step SSE_saw (steady-state error)

Metrics computed from stored raw realized latencies:
  - Causal MA(20)  for T_r, T_s, OS  (avoids non-causal artefacts)
  - Centered MA(30) for SSE           (matches paper tracking figures)

Usage:
    python -m step4_deploy.plot_hyperparameter_table \
        --out results/step4_deploy/control/hyperparameter_table.pdf
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

# ── configs to evaluate ───────────────────────────────────────────────────────
CONFIGS = [
    # (alpha, kp,  ki,    json_path,                                   label)
    ("0.83", "2.0", "0.33",
     OUTPUTS / "jetson_trtengine_720x1280_b0.83_kp2.0_ki0.33_win45.json",
     "under-smoothed"),
    ("0.85", "2.0", "0.33",
     OUTPUTS / "jetson_trtengine_720x1280_b0.85_kp2.0_ki0.33.json",
     "chosen"),
    ("0.93", "2.0", "0.33",
     OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp2.0_ki0.33.json",
     "over-smoothed"),
    ("0.93", "2.0", "2.0",
     OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp2.0_ki2.0.json",
     "high $k_i$"),
    ("0.93", "4.0", "0.80",
     OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp4.0_ki0.80.json",
     "high $k_p$"),
]

CHOSEN_ROW = 1   # index into CONFIGS

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


# ── metric computation ────────────────────────────────────────────────────────

def compute_metrics(path, cma_win=20, disp_win=30, pre=60, settle_band=1.5):
    d    = json.loads(Path(path).read_text())
    sse_step, sse_saw = [], []
    up_tr, up_ts, up_os = [], [], []
    dn_tr, dn_ts, dn_os = [], [], []

    for fam in d['fam_order']:
        for btype in ('step', 'sawtooth'):
            cell = d['cells'][f'{fam}/{btype}']
            raw  = np.asarray(cell['realized'], dtype=float)
            tgt  = np.asarray(cell['target'],   dtype=float)
            sm   = centered_ma(raw, disp_win)
            mae  = float(np.mean(np.abs(sm - tgt)))
            (sse_step if btype == 'step' else sse_saw).append(mae)

            if btype != 'step':
                continue

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

                os_ms = (max(0.0, post.max() - t_after) if delta > 0
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

    return dict(
        sse_step = float(np.mean(sse_step)),
        sse_saw  = float(np.mean(sse_saw)),
        tr_up    = m(up_tr),   ts_up = m(up_ts),   os_up = float(np.mean(up_os)),
        tr_dn    = m(dn_tr),   ts_dn = m(dn_ts),   os_dn = float(np.mean(dn_os)),
    )


# ── figure ────────────────────────────────────────────────────────────────────

def cell_color(val, lo, hi, good="low"):
    """
    Interpolate cell background: green (good) → white → red (bad).
    good='low'  → low values are green, high are red.
    good='high' → high values are green, low are red.
    """
    if lo == hi:
        return (1.0, 1.0, 1.0)
    t = (val - lo) / (hi - lo)     # 0 = best, 1 = worst
    if good == "high":
        t = 1.0 - t
    t = max(0.0, min(1.0, t))
    # green=(0.67,0.86,0.67) white=(1,1,1) red=(0.95,0.63,0.63)
    if t <= 0.5:
        f = t * 2
        return (0.67 + f * (1.0 - 0.67),
                0.86 + f * (1.0 - 0.86),
                0.67 + f * (1.0 - 0.67))
    else:
        f = (t - 0.5) * 2
        return (1.0 + f * (0.95 - 1.0),
                1.0 + f * (0.63 - 1.0),
                1.0 + f * (0.63 - 1.0))


def fmt(v, decimals=0, suffix=""):
    if v is None:
        return "—"
    if decimals == 0:
        return f"{int(v)}{suffix}"
    return f"{v:.{decimals}f}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",     default=str(OUTPUTS / "hyperparameter_table.pdf"))
    ap.add_argument("--cma",     type=int,   default=20)
    ap.add_argument("--win",     type=int,   default=30)
    ap.add_argument("--pre",     type=int,   default=60)
    ap.add_argument("--settle",  type=float, default=1.5)
    args = ap.parse_args()

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # ── compute all rows ──────────────────────────────────────────────────────
    rows = []
    for alpha, kp, ki, path, label in CONFIGS:
        m = compute_metrics(path, args.cma, args.win, args.pre, args.settle)
        rows.append((alpha, kp, ki, label, m))

    # ── table structure ───────────────────────────────────────────────────────
    # Columns:
    #  α | kp | ki || Tr↑ | Tr↓ || Ts↑ | Ts↓ || OS↑% | OS↓% || SSE_step | SSE_saw
    COL_HEADERS = [
        r"$\alpha$", r"$k_p$", r"$k_i$",
        r"$T_r^{\uparrow}$(f)", r"$T_r^{\downarrow}$(f)",
        r"$T_s^{\uparrow}$(f)", r"$T_s^{\downarrow}$(f)",
        r"OS$^{\uparrow}$(%)", r"OS$^{\downarrow}$(%)",
        "SSE\nstep (ms)", "SSE\nsaw. (ms)",
    ]

    # column groups for visual separator (0-indexed, separator AFTER this col)
    SEPARATORS_AFTER = {2, 4, 6, 8}

    def row_vals(m):
        return [
            fmt(m['tr_up']), fmt(m['tr_dn']),
            fmt(m['ts_up']), fmt(m['ts_dn']),
            fmt(m['os_up'], 0, ""),  fmt(m['os_dn'], 0, ""),
            fmt(m['sse_step'], 3),   fmt(m['sse_saw'], 3),
        ]

    table_data = []
    for alpha, kp, ki, label, m in rows:
        table_data.append([alpha, kp, ki] + row_vals(m))

    n_rows = len(table_data)
    n_cols = len(COL_HEADERS)

    # value ranges for coloring (only the numeric columns, indices 3..10)
    numeric_idx = list(range(3, n_cols))
    # all lower = better for everything in this table
    col_vals = {}
    for ci in numeric_idx:
        vals = []
        for row in table_data:
            try:
                vals.append(float(row[ci].replace("—", "nan")))
            except Exception:
                vals.append(np.nan)
        col_vals[ci] = np.array(vals, dtype=float)

    # ── matplotlib table ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")

    col_widths = [0.07, 0.06, 0.06,   # α, kp, ki
                  0.08, 0.08,           # Tr↑, Tr↓
                  0.08, 0.08,           # Ts↑, Ts↓
                  0.07, 0.07,           # OS↑, OS↓
                  0.10, 0.10]           # SSE_step, SSE_saw
    assert len(col_widths) == n_cols

    # Draw the table manually for full control
    row_h   = 0.115
    header_h= 0.140
    total_h = header_h + n_rows * row_h + 0.04
    x0 = 0.01; y0 = 0.98

    # cumulative x positions
    xs = [x0]
    for w in col_widths:
        xs.append(xs[-1] + w)
    x_total = xs[-1]

    # ── group header bars ─────────────────────────────────────────────────────
    GROUP_DEFS = [
        ("Config",      0,  2,  "#4A6FA5"),
        ("Rise time",   3,  4,  "#6B8E6B"),
        ("Settling",    5,  6,  "#8B6B8E"),
        ("Overshoot",   7,  8,  "#B07070"),
        ("Steady-state error", 9, 10, "#7F8C8D"),
    ]
    grp_y = y0 - 0.03
    grp_h = 0.06
    for grp_lbl, c0, c1, color in GROUP_DEFS:
        gx0 = xs[c0]; gx1 = xs[c1 + 1]
        rect = plt.Rectangle((gx0 + 0.002, grp_y - grp_h),
                              gx1 - gx0 - 0.004, grp_h,
                              transform=ax.transAxes,
                              facecolor=color, edgecolor="white", linewidth=0.8)
        ax.add_patch(rect)
        ax.text((gx0 + gx1) / 2, grp_y - grp_h / 2, grp_lbl,
                transform=ax.transAxes,
                ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold")

    # ── column headers ────────────────────────────────────────────────────────
    hdr_y = grp_y - grp_h
    hdr_bg = "#D0D0D0"
    for ci, (hdr, w, xc) in enumerate(zip(COL_HEADERS, col_widths,
                                           [0.5*(xs[i]+xs[i+1]) for i in range(n_cols)])):
        rect = plt.Rectangle((xs[ci] + 0.001, hdr_y - row_h + 0.002),
                              xs[ci+1] - xs[ci] - 0.002, row_h - 0.004,
                              transform=ax.transAxes,
                              facecolor=hdr_bg, edgecolor="white", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(xc, hdr_y - row_h / 2, hdr,
                transform=ax.transAxes,
                ha="center", va="center", fontsize=5.8, color="#222222",
                fontweight="bold", linespacing=1.2)

    # ── data rows ─────────────────────────────────────────────────────────────
    for ri, (row_vals_i, (alpha, kp, ki, label, m)) in enumerate(
            zip(table_data, rows)):
        is_chosen = (ri == CHOSEN_ROW)
        row_y_top = hdr_y - row_h - ri * row_h

        for ci, val in enumerate(row_vals_i):
            cx = 0.5 * (xs[ci] + xs[ci + 1])
            cy = row_y_top - row_h / 2

            # Background color
            if is_chosen:
                bg = "#FFF9C4"          # soft yellow highlight
            elif ci in numeric_idx:
                v_arr = col_vals[ci]
                finite = v_arr[np.isfinite(v_arr)]
                if len(finite) >= 2:
                    try:
                        fv = float(val.replace("—", "nan"))
                        bg = cell_color(fv, finite.min(), finite.max(), good="low")
                    except Exception:
                        bg = (1.0, 1.0, 1.0)
                else:
                    bg = (1.0, 1.0, 1.0)
            else:
                bg = "#F5F5F5" if ri % 2 == 0 else (1.0, 1.0, 1.0)

            rect = plt.Rectangle((xs[ci] + 0.001, row_y_top - row_h + 0.002),
                                  xs[ci+1] - xs[ci] - 0.002, row_h - 0.004,
                                  transform=ax.transAxes,
                                  facecolor=bg, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)

            txt = val
            fw  = "bold" if is_chosen else "normal"
            fs  = 6.5
            ax.text(cx, cy, txt, transform=ax.transAxes,
                    ha="center", va="center", fontsize=fs,
                    fontweight=fw, color="#111111")

        # chosen star in margin
        if is_chosen:
            ax.text(xs[0] - 0.008, row_y_top - row_h / 2, "★",
                    transform=ax.transAxes,
                    ha="right", va="center", fontsize=7, color="#B8860B")

        # row label (right margin)
        ax.text(xs[-1] + 0.005, row_y_top - row_h / 2, label,
                transform=ax.transAxes,
                ha="left", va="center", fontsize=5.8,
                color="#555555" if not is_chosen else "#B8860B",
                style="italic" if not is_chosen else "normal",
                fontweight="bold" if is_chosen else "normal")

    # ── outer border ─────────────────────────────────────────────────────────
    total_table_h = grp_h + row_h + n_rows * row_h
    border = plt.Rectangle((x0, grp_y - total_table_h - 0.005),
                            x_total - x0, total_table_h + 0.005,
                            transform=ax.transAxes,
                            facecolor="none", edgecolor="#888888", linewidth=0.8)
    ax.add_patch(border)

    # ── vertical separators (after groups) ───────────────────────────────────
    for sep_ci in SEPARATORS_AFTER:
        sx = xs[sep_ci + 1]
        ax.plot([sx, sx], [grp_y - total_table_h, grp_y],
                transform=ax.transAxes,
                color="#888888", lw=0.8, ls="--")

    # ── caption / footnote ────────────────────────────────────────────────────
    footnote_y = grp_y - total_table_h - 0.08
    ax.text(x0, footnote_y,
            r"Metrics: $T_r$ = rise time (90% of step, causal MA-20), "
            r"$T_s$ = settling (±1.5 ms, causal MA-20), "
            r"OS = overshoot, SSE = mean $|$realized$-$target$|$ (centered MA-30). "
            r"↑ = step-up budget, ↓ = step-down. "
            r"★ = chosen default ($\alpha$=0.85, $k_p$=2.0, $k_i$=0.33).",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=5.2, color="0.40", wrap=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(f"[*] -> {out}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print("\n% ── LaTeX table ─────────────────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{PI controller hyperparameter sensitivity on Jetson Orin Nano "
          r"(BDD100K, 720$\times$1280). "
          r"$T_r$: rise time (90\% of step); $T_s$: settling time ($\pm$1.5\,ms); "
          r"OS: overshoot; SSE: mean $|$realized\,$-$\,target$|$. "
          r"$\uparrow$/$\downarrow$: step-up/down budget. "
          r"\textbf{Bold}: chosen default.}")
    print(r"\label{tab:pi_sensitivity}")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\renewcommand{\arraystretch}{1.1}")
    print(r"\begin{tabular}{ccc|cc|cc|cc|cc}")
    print(r"\hline")
    print(r"\multicolumn{3}{c|}{Config} & "
          r"\multicolumn{2}{c|}{$T_r$ (f)} & "
          r"\multicolumn{2}{c|}{$T_s$ (f)} & "
          r"\multicolumn{2}{c|}{OS (\%)} & "
          r"\multicolumn{2}{c}{SSE (ms)} \\")
    print(r"$\alpha$ & $k_p$ & $k_i$ & "
          r"$\uparrow$ & $\downarrow$ & "
          r"$\uparrow$ & $\downarrow$ & "
          r"$\uparrow$ & $\downarrow$ & "
          r"step & saw. \\")
    print(r"\hline")
    for ri, (alpha, kp, ki, label, m) in enumerate(rows):
        def f(v, d=0): return fmt(v, d)
        cells = [alpha, kp, ki,
                 f(m['tr_up']), f(m['tr_dn']),
                 f(m['ts_up']), f(m['ts_dn']),
                 f(m['os_up']), f(m['os_dn']),
                 f(m['sse_step'], 3), f(m['sse_saw'], 3)]
        if ri == CHOSEN_ROW:
            cells = [r"\textbf{" + c + "}" for c in cells]
        row_str = " & ".join(cells) + r" \\"
        if ri == CHOSEN_ROW:
            row_str += "  % chosen"
        print(row_str)
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n{'α':>5} {'kp':>5} {'ki':>5}  "
          f"{'Tr↑':>5} {'Tr↓':>5}  {'Ts↑':>5} {'Ts↓':>5}  "
          f"{'OS↑':>6} {'OS↓':>6}  {'SSE-step':>9} {'SSE-saw':>9}")
    print("-" * 80)
    for alpha, kp, ki, label, m in rows:
        star = "★" if label == "chosen" else " "
        print(f"{star}{alpha:>4} {kp:>5} {ki:>5}  "
              f"{fmt(m['tr_up']):>5} {fmt(m['tr_dn']):>5}  "
              f"{fmt(m['ts_up']):>5} {fmt(m['ts_dn']):>5}  "
              f"{m['os_up']:>5.0f}% {m['os_dn']:>5.0f}%  "
              f"{m['sse_step']:>9.3f} {m['sse_saw']:>9.3f}   ({label})")


if __name__ == "__main__":
    main()
