"""PI controller hyperparameter ablation figure.

Groups existing experimental runs to show:
  (a) sensitivity to beta (EMA smoothing coefficient) — with fixed kp=2.0, ki=0.33
  (b) sensitivity to gains (kp, ki) — with fixed beta=0.93 (worst-case beta)

Demonstrates that beta is the dominant parameter and that within-range gain variation
causes only modest (<0.1 ms) changes in tracking MAE.

Usage:
    python -m step4_deploy.plot_ablation \
        --out results/step4_deploy/control/ablation_controller.pdf
"""
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "bdd100k"

JSONS = {
    "b0.83_kp2.0_ki0.33": OUTPUTS / "jetson_trtengine_720x1280_b0.83_kp2.0_ki0.33_win45.json",
    "b0.85_kp2.0_ki0.33": OUTPUTS / "jetson_trtengine_720x1280_b0.85_kp2.0_ki0.33.json",
    "b0.93_kp2.0_ki0.33": OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp2.0_ki0.33.json",
    "b0.93_kp2.0_ki2.0":  OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp2.0_ki2.0.json",
    "b0.93_kp4.0_ki0.80": OUTPUTS / "jetson_trtengine_720x1280_b0.93_kp4.0_ki0.80.json",
}

CHOSEN = "b0.85_kp2.0_ki0.33"


def trail(x, n):
    pad = n // 2
    x_padded = np.pad(x, (pad, n - pad - 1), mode='edge')
    return np.convolve(x_padded, np.ones(n) / n, mode='valid')


def load_maes(path, win=30):
    d = json.loads(Path(path).read_text())
    step_maes, saw_maes = [], []
    for k, cell in d["cells"].items():
        sm = trail(np.asarray(cell["realized"]), win)
        tgt = np.asarray(cell["target"])
        mae = np.mean(np.abs(sm - tgt))
        (step_maes if "step" in k else saw_maes).append(mae)
    return np.mean(step_maes), np.mean(saw_maes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUTS / "ablation_controller.pdf"))
    ap.add_argument("--win", type=int, default=30)
    args = ap.parse_args()

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # ── Data ──────────────────────────────────────────────────────────────────
    data = {}
    for name, path in JSONS.items():
        data[name] = load_maes(path, args.win)

    # Split into two groups for the two sub-panels
    # (a) beta sensitivity — same kp/ki, vary beta
    beta_group = [
        (r"$\beta=0.83$", "b0.83_kp2.0_ki0.33"),
        (r"$\beta=0.85$", "b0.85_kp2.0_ki0.33"),   # chosen
        (r"$\beta=0.93$", "b0.93_kp2.0_ki0.33"),
    ]
    # (b) gain sensitivity — fixed (suboptimal) beta=0.93, vary kp/ki
    gain_group = [
        (r"$k_p{=}2.0,\,k_i{=}0.33$", "b0.93_kp2.0_ki0.33"),
        (r"$k_p{=}2.0,\,k_i{=}2.0$",  "b0.93_kp2.0_ki2.0"),
        (r"$k_p{=}4.0,\,k_i{=}0.80$", "b0.93_kp4.0_ki0.80"),
    ]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4),
                             gridspec_kw={"wspace": 0.38})

    C_STEP = "#4878CF"
    C_SAW  = "#D65F5F"
    W = 0.32
    ALPHA_DEFAULT = 1.0
    ALPHA_OTHER   = 0.55

    for ax, group, title, note in [
        (axes[0], beta_group,
         r"(a) $\beta$ sensitivity  ($k_p{=}2.0,\,k_i{=}0.33$)",
         r"$\uparrow\beta$ over-smooths target$\Rightarrow$slow sawtooth response"),
        (axes[1], gain_group,
         r"(b) Gain sensitivity  ($\beta{=}0.93$)",
         r"Gain changes cause only $\leq$0.1 ms shift"),
    ]:
        xs = np.arange(len(group))
        for i, (label, key) in enumerate(group):
            step_mae, saw_mae = data[key]
            is_chosen = (key == CHOSEN)
            alpha = ALPHA_DEFAULT if not is_chosen else ALPHA_DEFAULT
            lw = 1.5 if is_chosen else 0.6
            ec_step = "black" if is_chosen else C_STEP
            ec_saw  = "black" if is_chosen else C_SAW

            b1 = ax.bar(xs[i] - W/2, step_mae, W, color=C_STEP,
                        alpha=alpha, linewidth=lw, edgecolor=ec_step,
                        label="Step" if i == 0 else "_")
            b2 = ax.bar(xs[i] + W/2, saw_mae,  W, color=C_SAW,
                        alpha=alpha, linewidth=lw, edgecolor=ec_saw,
                        label="Sawtooth" if i == 0 else "_")

            if is_chosen:
                ax.bar(xs[i] - W/2, step_mae, W, color="none", linewidth=1.8,
                       edgecolor="black")
                ax.bar(xs[i] + W/2, saw_mae,  W, color="none", linewidth=1.8,
                       edgecolor="black")
                ax.text(xs[i], -0.12, "★", ha="center", va="top",
                        fontsize=8, color="black",
                        transform=ax.get_xaxis_transform())

        ax.set_xticks(xs)
        ax.set_xticklabels([g[0] for g in group], fontsize=6.5)
        ax.set_ylabel("MAE (ms)", fontsize=8)
        ax.set_ylim(0, 1.25)
        ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
        ax.set_title(title, fontsize=7.5, pad=4)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="y", alpha=0.25, ls="--")
        ax.text(0.97, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=5.8, color="0.35",
                style="italic")

    # shared legend
    handles = [
        mpatches.Patch(color=C_STEP, label="Step budget"),
        mpatches.Patch(color=C_SAW,  label="Sawtooth budget"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))

    fig.text(0.5, -0.13,
             r"$\star$ denotes the chosen default ($\beta{=}0.85,\,k_p{=}2.0,\,k_i{=}0.33$); "
             r"all MAEs computed with a $w{=}30$-frame moving average.",
             ha="center", va="top", fontsize=6.0, color="0.4")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    print(f"[*] -> {out}")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'Config':<28} {'Step MAE':>10} {'Saw MAE':>10} {'Mean MAE':>10}")
    print("-" * 62)
    for name, (s, w) in data.items():
        marker = " ← chosen" if name == CHOSEN else ""
        print(f"{name:<28} {s:>9.3f}ms {w:>9.3f}ms {(s+w)/2:>9.3f}ms{marker}")


if __name__ == "__main__":
    main()
