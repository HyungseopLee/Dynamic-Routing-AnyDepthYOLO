"""Motivation figure: why regress the loss-advantage (gap) as the routing target.

Two claims, one panel each, on both datasets (rows = KITTI, BDD100K):
  (a) per-image detection loss is a strong proxy for per-image AP@[.50:.95]
      -- shown for BOTH paths (BASE red, SUPER blue);
  (b) the per-image advantage A = L_base - L_super predicts the realized AP gain
      of the deep path (AP_super - AP_base), i.e. the scalar the router regresses
      actually ranks frames by how much SUPER helps.

Reads the precomputed per-image tables produced by tools/analyze_loss_pr.py:
    anydepth-yolov12/analysis/{kitti,bdd100k}-AnyDepth/per_image_loss_pr_conf.csv
Writes paper/_extracted/fig_loss_ap.pdf.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent / "anydepth-yolov12" / "analysis"
OUT = Path(__file__).resolve().parent / "_extracted" / "fig_loss_ap.pdf"
DATASETS = [("KITTI", ROOT / "kitti-AnyDepth/per_image_loss_pr_conf.csv"),
            ("BDD100K", ROOT / "bdd100k-AnyDepth/per_image_loss_pr_conf.csv")]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})


def corr_txt(x, y):
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    return r, rho


def main():
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(7.0, 5.2))
    for row, (dname, csv) in enumerate(DATASETS):
        df = pd.read_csv(csv)
        # ---- (a) loss vs AP, both paths -----------------------------------
        ax = axes[row][0]
        for tag, c, lbl in (("base", "tab:red", "BASE"),
                            ("super", "tab:blue", "SUPER")):
            L = df[f"loss_{tag}"].to_numpy()
            ap = df[f"ap5095_{tag}"].to_numpy()
            r, rho = corr_txt(L, ap)
            ax.scatter(L, ap, s=2, c=c, alpha=0.18, edgecolors="none")
            # linear fit line
            b1, b0 = np.polyfit(L, ap, 1)
            xx = np.linspace(L.min(), L.max(), 50)
            ax.plot(xx, b0 + b1 * xx, c=c, lw=1.4,
                    label=f"{lbl}: $r$={r:.2f}, $\\rho$={rho:.2f}")
        ax.set_xlabel("per-image detection loss", fontsize=8.5)
        ax.set_ylabel("AP$_{50:95}$", fontsize=8.5)
        ax.set_title(f"({chr(97+row*2)}) {dname}: loss vs. AP", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
        ax.grid(alpha=0.22, ls="--"); ax.tick_params(labelsize=7.5)

        # ---- (b) advantage vs AP gap --------------------------------------
        ax = axes[row][1]
        A = (df["loss_base"] - df["loss_super"]).to_numpy()          # advantage
        apgap = (df["ap5095_super"] - df["ap5095_base"]).to_numpy()  # deep-path gain
        r, rho = corr_txt(A, apgap)
        ax.scatter(A, apgap, s=2, c="tab:purple", alpha=0.14, edgecolors="none")
        # binned mean to make the (noisy per-image) trend legible
        qe = np.quantile(A, np.linspace(0.01, 0.99, 13))
        bx, by = [], []
        for i in range(len(qe) - 1):
            m = (A >= qe[i]) & (A < qe[i + 1])
            if m.sum() > 5:
                bx.append(A[m].mean()); by.append(apgap[m].mean())
        ax.plot(bx, by, "o", c="tab:purple", ms=4, zorder=6)
        b1, b0 = np.polyfit(A, apgap, 1)
        xx = np.linspace(np.quantile(A, 0.005), np.quantile(A, 0.995), 50)
        ax.plot(xx, b0 + b1 * xx, c="black", lw=1.4,
                label=f"$r$={r:.2f}, $\\rho$={rho:.2f}")
        ax.axhline(0, color="0.6", lw=0.7, ls="--")
        ax.axvline(0, color="0.6", lw=0.7, ls="--")
        lo, hi = np.quantile(A, [0.01, 0.99])
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"advantage $A=L_{\mathrm{base}}-L_{\mathrm{super}}$",
                      fontsize=8.5)
        ax.set_ylabel(r"AP gap (SUPER $-$ BASE)", fontsize=8.5)
        ax.set_title(f"({chr(97+row*2+1)}) {dname}: advantage vs. AP gain",
                     fontsize=9)
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
        ax.grid(alpha=0.22, ls="--"); ax.tick_params(labelsize=7.5)
        print(f"[{dname}] loss-AP base r={corr_txt(df.loss_base, df.ap5095_base)[0]:.3f} "
              f"super r={corr_txt(df.loss_super, df.ap5095_super)[0]:.3f} | "
              f"advantage-APgap r={r:.3f} rho={rho:.3f}")

    fig.tight_layout(pad=0.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"[*] -> {OUT}")


if __name__ == "__main__":
    main()
