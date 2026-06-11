"""Advantage-distribution figure: why the useful tau band is narrow on BDD100K.

Overlays the per-image advantage A = L_base - L_super for KITTI and BDD100K. BDD's
distribution is concentrated near zero (smaller base->super gap), so the whole
base<->super operating range is reached within a narrow tau band; KITTI's is wider.
Reads the per-image tables from tools/analyze_loss_pr.py.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "anydepth-yolov12" / "analysis"
OUT = Path(__file__).resolve().parent / "_extracted" / "fig_advantage_dist.pdf"
SRC = [("KITTI", "tab:red", ROOT / "kitti-AnyDepth/per_image_loss_pr_conf.csv"),
       ("BDD100K", "tab:blue", ROOT / "bdd100k-AnyDepth/per_image_loss_pr_conf.csv")]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(figsize=(3.45, 2.65))
bins = np.linspace(-0.6, 0.7, 80)
for name, c, csv in SRC:
    d = pd.read_csv(csv)
    A = (d["loss_base"] - d["loss_super"]).to_numpy()
    ax.hist(A, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=c)
    ax.hist(A, bins=bins, density=True, histtype="step", lw=1.6, color=c,
            label=f"{name} ($\\sigma$={A.std():.2f})")
ax.axvline(0, color="0.5", lw=0.9, ls=":")
ax.set_xlabel(r"advantage $A=L_{\mathrm{base}}-L_{\mathrm{super}}$", fontsize=9)
ax.set_ylabel("density", fontsize=9)
ax.tick_params(labelsize=8)
ax.set_xlim(-0.6, 0.7)
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.grid(alpha=0.22, ls="--")
fig.tight_layout(pad=0.3)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print(f"[*] -> {OUT}")
