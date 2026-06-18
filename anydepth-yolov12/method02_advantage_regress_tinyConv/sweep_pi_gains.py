"""PI gain sensitivity: how robust is budget tracking to the controller gains?

Sweeps the proportional/integral gains (kp, ki) of the latency-budget controller
over a grid and reports, for each, the budget-tracking MAE averaged over the step
and sawtooth schedules. A flat low-MAE plateau means the method is not delicate
about gain tuning. Reuses simulate()/target_schedule()/smooth() from the main sim.

    conda run -n yolov12 python -m method02_advantage_regress_tinyConv.sweep_pi_gains
"""

import pickle
from pathlib import Path

import numpy as np

from method02_advantage_regress_tinyConv.sim_latency_budget import (
    simulate, target_schedule, smooth, L_BASE, L_SUPER, L_ROUTER)

KP_GRID = [0.005, 0.010, 0.020, 0.040, 0.080]
KI_GRID = [0.001, 0.002, 0.004, 0.008, 0.016]
DEFAULT = (0.020, 0.004)  # gains used in the main Figure 6


def track_metrics(frames, kp, ki):
    """Mean over step+sawtooth of (MAE, max over-budget overshoot)."""
    n = len(frames)
    maes, overs = [], []
    for kind in ("step", "sawtooth"):
        Ltgt = target_schedule(kind, n)
        res = simulate(frames, Ltgt, "pi", kp=kp, ki=ki, score=False)
        sm = smooth(res["realized"])
        maes.append(float(np.mean(np.abs(sm - Ltgt))))
        overs.append(float(np.max(np.clip(sm - Ltgt, 0, None))))
    return float(np.mean(maes)), float(np.mean(overs))


def main():
    dump = Path(__file__).resolve().parent / "outputs/kitti/perframe_dump.pkl"
    frames = pickle.load(open(dump, "rb"))["frames"]
    out = (Path(__file__).resolve().parent.parent.parent /
           "paper/_extracted/fig_pi_gain_sweep.pdf")

    mae = np.zeros((len(KI_GRID), len(KP_GRID)))
    over = np.zeros_like(mae)
    for r, ki in enumerate(KI_GRID):
        for c, kp in enumerate(KP_GRID):
            mae[r, c], over[r, c] = track_metrics(frames, kp, ki)
            print(f"kp={kp:.3f} ki={ki:.3f}  MAE={mae[r,c]:.3f}  "
                  f"max_over={over[r,c]:.3f}")

    print(f"\nMAE range [{mae.min():.3f}, {mae.max():.3f}] ms over the grid")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    im = ax.imshow(mae, cmap="viridis_r", origin="lower", aspect="auto")
    ax.set_xticks(range(len(KP_GRID)))
    ax.set_xticklabels([f"{v:g}" for v in KP_GRID], fontsize=8)
    ax.set_yticks(range(len(KI_GRID)))
    ax.set_yticklabels([f"{v:g}" for v in KI_GRID], fontsize=8)
    ax.set_xlabel(r"proportional gain $K_p$", fontsize=9)
    ax.set_ylabel(r"integral gain $K_i$", fontsize=9)
    for r in range(len(KI_GRID)):
        for c in range(len(KP_GRID)):
            ax.text(c, r, f"{mae[r,c]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if mae[r, c] > mae.mean() else "black")
    # mark the default gains used in Figure 6
    dc, dr = KP_GRID.index(DEFAULT[0]), KI_GRID.index(DEFAULT[1])
    ax.add_patch(plt.Rectangle((dc - 0.5, dr - 0.5), 1, 1, fill=False,
                               edgecolor="tab:red", lw=2.0))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("tracking MAE (ms)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] -> {out}")


if __name__ == "__main__":
    main()
