"""Combine latency-mode and energy-mode budget-tracking figures into one compact PDF.

Layout: 3 rows (families) × 4 cols (latency-step | latency-saw | energy-step | energy-saw).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_deploy.online_budget_demo import trail, SHORT, TITLES


def render_combined(lat_json, eng_json, out, win_lat=None, win_eng=None):
    lat = json.loads(Path(lat_json).read_text())
    eng = json.loads(Path(eng_json).read_text())

    win_lat = win_lat or lat.get("win", 45)
    win_eng = win_eng or eng.get("win", 30)

    fams = lat["fam_order"]

    # Latency y-range from target values
    all_lat_tgt = np.concatenate([np.asarray(lat["cells"][f"{f}/{b}"]["target"])
                                   for f in fams for b in ("step", "sawtooth")])
    lat_ylo = all_lat_tgt.min() - 1.5
    lat_yhi = all_lat_tgt.max() + 2.5

    # Energy y-range
    e_base = eng["e_base"]; e_super = eng["e_super"]
    eng_ylo = e_base - 30;  eng_yhi = e_super + 60

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})

    n_rows = len(fams)
    fig, axes = plt.subplots(n_rows, 4, figsize=(7.2, 1.65 * n_rows),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.35})

    col_titles = ["Latency – Step", "Latency – Sawtooth",
                  "Energy – Step",  "Energy – Sawtooth"]
    budgets    = ["step", "sawtooth", "step", "sawtooth"]
    modes      = ["latency", "latency", "energy", "energy"]

    print(f"\n{'scenario':<14}{'mode':<10}{'budget':<12}{'MAE':>10}{'SUPER%':>9}")

    for r, fam in enumerate(fams):
        bounds = lat["families"][fam]["bounds"]
        labels = lat["families"][fam]["labels"]

        for c, (bkind, mode) in enumerate(zip(budgets, modes)):
            ax = axes[r][c]
            dump = lat if mode == "latency" else eng
            win  = win_lat if mode == "latency" else win_eng
            ylo  = lat_ylo if mode == "latency" else eng_ylo
            yhi  = lat_yhi if mode == "latency" else eng_yhi

            cell = dump["cells"][f"{fam}/{bkind}"]
            realized = np.asarray(cell["realized"])
            tgt      = np.asarray(cell["target"])
            n        = len(realized)
            sm       = trail(realized, win)

            if mode == "latency":
                mae_str = f"{np.mean(np.abs(sm - tgt)):.2f} ms"
            else:
                mae_str = f"{np.mean(np.abs(sm - tgt)):.1f} mJ"
            mae_val = float(np.mean(np.abs(sm - tgt)))
            unit    = "ms" if mode == "latency" else "mJ"
            print(f"{fam:<14}{mode:<10}{bkind:<12}{mae_val:>9.2f} {unit}{cell['super_rate']*100:>8.1f}%")

            # Condition shading
            for k in range(len(labels)):
                x0, x1 = bounds[k], bounds[k + 1]
                ax.axvspan(x0, x1,
                           color="tab:blue" if k % 2 == 0 else "tab:orange",
                           alpha=0.07, zorder=0)
                if r == 0:
                    ax.text((x0 + x1) / 2, yhi - (yhi - ylo) * 0.04,
                            SHORT.get(labels[k], labels[k]),
                            ha="center", va="top", fontsize=5.5, color="0.4")
                if k > 0:
                    ax.axvline(x0, color="0.6", ls="-", lw=0.5, alpha=0.4, zorder=1)

            ax.plot(tgt, color="black", ls="--", lw=1.1, zorder=6)
            ax.plot(sm,  color="tab:red", lw=1.3, zorder=5)
            ax.set_ylim(ylo, yhi); ax.set_xlim(0, n)
            ax.grid(alpha=0.18, ls="--"); ax.tick_params(labelsize=6)

            # MAE annotation
            ax.text(0.03, 0.06, f"MAE={mae_str}", transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.85))

            # Column title (top row only)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8, pad=3)

            # y-axis label (leftmost latency col + leftmost energy col)
            if c == 0:
                ax.set_ylabel(f"{TITLES.get(fam, fam)}\nlatency (ms)", fontsize=6.5)
            elif c == 2:
                ax.set_ylabel("energy (mJ)", fontsize=6.5)
            else:
                ax.set_yticklabels([])

            # x-axis label (bottom row only)
            if r == n_rows - 1:
                ax.set_xlabel("frame", fontsize=7)

            # Vertical separator between the two modes
            if c == 1:
                ax.spines["right"].set_visible(True)
                ax.spines["right"].set_linewidth(0.8)
                ax.spines["right"].set_color("0.5")

    # Shared legend at bottom
    handles = [
        Line2D([0], [0], color="black", ls="--", lw=1.1, label="target budget $B^\\star(t)$"),
        Line2D([0], [0], color="tab:red",  lw=1.3,       label="realized (smoothed)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(f"\n[*] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat",  required=True, help="latency-mode JSON dump")
    ap.add_argument("--eng",  required=True, help="energy-mode JSON dump")
    ap.add_argument("--out",  required=True, help="output PDF path")
    ap.add_argument("--win_lat", type=int, default=None)
    ap.add_argument("--win_eng", type=int, default=None)
    args = ap.parse_args()
    render_combined(args.lat, args.eng, args.out, args.win_lat, args.win_eng)


if __name__ == "__main__":
    main()
