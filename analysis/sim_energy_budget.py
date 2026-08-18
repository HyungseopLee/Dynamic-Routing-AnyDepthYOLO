"""Target-budget scheduling with PI control (energy or FPS mode).

Replays perframe_scenarios.pkl under a time-varying budget target and adjusts
the router threshold tau each frame so realized cost tracks the budget.

Energy model: E_frame = L_frame × DEVICE_POWER_W  (mJ per frame)
  Anchors (Jetson Orin Nano, 720×1280, TRT FP16):
    L_BASE  = 25.12 ms  → E_BASE  = 25.12 × 15 W ≈ 376.8 mJ
    L_SUPER = 42.12 ms  → E_SUPER = 42.12 × 15 W ≈ 631.8 mJ
    FPS_BASE  = 1000 / (L_BASE  + L_ROUTER) ≈ 38.3 fps
    FPS_SUPER = 1000 / (L_SUPER + L_ROUTER) ≈ 23.2 fps

Usage:
    # energy budget figure
    python -m analysis.sim_energy_budget \
        --mode energy \
        --dump results/step3_eval/bdd100k/perframe_scenarios.pkl \
        --out  results/figures/fig_energy_budget_bdd.pdf

    # FPS budget figure
    python -m analysis.sim_energy_budget \
        --mode fps \
        --dump results/step3_eval/bdd100k/perframe_scenarios.pkl \
        --out  results/figures/fig_fps_budget_bdd.pdf
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_IOU = 10   # thresholds 0.50, 0.55, ..., 0.95


def _ap_at_iou(matches_by_class, gt_count, iou_idx):
    """Compute AP at a single IoU threshold index using 101-point interpolation."""
    aps = []
    for cls, mlist in matches_by_class.items():
        n_gt = gt_count.get(cls, 0)
        if n_gt == 0:
            continue
        mlist.sort(key=lambda x: -x[1])   # sort by confidence desc
        tp = np.array([m[2][iou_idx] for m in mlist], dtype=float)
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        prec = tp_cum / (tp_cum + fp_cum + 1e-9)
        rec  = tp_cum / (n_gt + 1e-9)
        # 101-point interpolation
        rec_interp = np.linspace(0, 1, 101)
        prec_interp = np.zeros(101)
        for j in range(len(prec) - 1, -1, -1):
            prec_interp[rec_interp <= rec[j]] = prec[j]
        aps.append(prec_interp.mean())
    return float(np.mean(aps)) if aps else 0.0


def compute_map(matches, gt_count):
    """matches: list of (cls, conf, [bool×10]). Returns mAP@50:95."""
    by_cls = {}
    for cls, conf, hits in matches:
        by_cls.setdefault(cls, []).append((cls, conf, hits))
    ap_per_iou = [_ap_at_iou(by_cls, gt_count, i) for i in range(N_IOU)]
    return float(np.mean(ap_per_iou))


# ── RTX 3090 PyTorch anchors (bench_device_bdd.py, 720×1280 BDD100K) ──────────
L_BASE   = 13.97   # ms  (measured)
L_SUPER  = 20.94   # ms  (measured)
L_ROUTER = 0.217   # ms  (measured)

E_BASE   = 4086.1  # mJ  (measured, pynvml)
E_SUPER  = 6975.3  # mJ  (measured, pynvml)
E_ROUTER = E_BASE / L_BASE * L_ROUTER  # proportional estimate (~63 mJ)

# ── Scene selection: pick representative sequences ─────────────────────────────
SCENE_ORDER = [
    # (seq_id,          label,            color)
    ("b1cebfb7-284f5117", "highway / day",   "#4e79a7"),   # highway clear day
    ("b1ca2e5d-84cf9134", "city / night",    "#f28e2b"),   # city clear night
    ("b1cac6a7-04e33135", "city / rainy",    "#59a14f"),   # city rainy day
    ("b1d0091f-75824d0d", "highway / rainy", "#e15759"),   # highway rainy day
    ("b1ceb32e-51852abe", "city / dawn",     "#b07aa1"),   # city dawn/dusk
    ("b1d10d08-c35503b8", "city / day",      "#76b7b2"),   # city clear day
]


FPS_BASE  = 1000.0 / (L_BASE  + L_ROUTER)   # ≈ 71.1 fps
FPS_SUPER = 1000.0 / (L_SUPER + L_ROUTER)   # ≈ 47.1 fps


def _step_target(n: int, lo: float, hi: float, fracs) -> np.ndarray:
    edges = np.round(np.linspace(0, n, len(fracs) + 1)).astype(int)
    out = np.empty(n)
    for i, f in enumerate(fracs):
        out[edges[i]:edges[i + 1]] = lo + f * (hi - lo)
    return out


def energy_target(n: int) -> np.ndarray:
    """E*(t): step function within feasible energy band [E_BASE+R, E_SUPER+R]."""
    lo = E_BASE  + E_ROUTER + 5.0
    hi = E_SUPER + E_ROUTER - 5.0
    fracs = [0.85, 0.40, 0.70, 0.15, 0.55, 0.30]
    return _step_target(n, lo, hi, fracs)


def fps_target(n: int) -> np.ndarray:
    """FPS*(t): step function within feasible FPS band [FPS_SUPER, FPS_BASE].
    Higher FPS → shorter latency budget → less energy per frame."""
    lo = FPS_SUPER + 0.5   # floor (≈ 23.7 fps, near always-super)
    hi = FPS_BASE  - 0.5   # ceiling (≈ 37.8 fps, near always-base)
    # Note: high FPS = less compute = more base; fracs map hi→base, lo→super
    fracs = [0.15, 0.80, 0.35, 0.90, 0.50, 0.70]
    return _step_target(n, lo, hi, fracs)


def smooth(x: np.ndarray, w: int = 60) -> np.ndarray:
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="same")[pad:pad + len(x)]


def compute_metrics(sm: np.ndarray, tgt: np.ndarray,
                    fps: float = 10.0, band: float = None) -> dict:
    """Compute standard PI control tracking metrics.

    band: full range of the tracked signal (used for 5% settle threshold).
          Defaults to energy band (E_SUPER - E_BASE).
    """
    if band is None:
        band = (E_SUPER + E_ROUTER) - (E_BASE + E_ROUTER)
    err = sm - tgt
    mae        = float(np.mean(np.abs(err)))
    max_over   = float(np.max(np.maximum(err,  0)))
    max_under  = float(np.max(np.maximum(-err, 0)))
    # settling time: frames until |err| < 5% of band after each step
    thresh = 0.05 * band
    # find step indices (target changes)
    steps = np.where(np.abs(np.diff(tgt)) > 1.0)[0] + 1
    settle_frames = []
    for s in steps:
        window = err[s:s + 300]
        settled = np.where(np.abs(window) < thresh)[0]
        if len(settled):
            settle_frames.append(int(settled[0]))
    avg_settle_s = float(np.mean(settle_frames) / fps) if settle_frames else float("nan")
    return {
        "mae":       mae,
        "overshoot": max_over,
        "undershoot": max_under,
        "settle_s":  avg_settle_s,
    }


def simulate_fixed_tau(all_frames, tau=0.15):
    """Fixed-tau routing (no PI control) — matches paper's reported AP performance."""
    n = len(all_frames)
    realized = np.empty(n)
    is_super = np.zeros(n, bool)
    matches_ours   = []
    matches_single = []
    gt_count_all   = {}
    prev_choice = None

    for i, (fr, first) in enumerate(all_frames):
        if first:
            prev_choice = None
        if prev_choice is None:
            choice = "base"
        else:
            pv = fr["ahat_super"] if prev_choice == "super" else fr["ahat_base"]
            choice = "super" if pv > tau else "base"
        prev_choice = choice

        E_frame = (E_SUPER if choice == "super" else E_BASE) + E_ROUTER
        realized[i] = E_frame
        is_super[i] = choice == "super"
        matches_ours.extend(fr[f"match_{choice}"])
        matches_single.extend(fr["match_super"])
        for c, k in fr["gt_count"].items():
            gt_count_all[c] = gt_count_all.get(c, 0) + k

    map_ours   = compute_map(matches_ours,   gt_count_all)
    map_single = compute_map(matches_single, gt_count_all)
    return {
        "realized":   realized,
        "is_super":   is_super,
        "super_rate": is_super.sum() / n,
        "map_ours":   map_ours,
        "map_single": map_single,
        "matches_ours":   matches_ours,
        "matches_single": matches_single,
        "gt_count":       gt_count_all,
    }


def find_best_tau(all_frames, tau_range=None):
    """Sweep tau to find the operating point with minimum AP drop."""
    if tau_range is None:
        tau_range = np.linspace(-0.2, 0.8, 41)
    best_tau, best_drop, best_res = None, float("inf"), None
    for tau in tau_range:
        res = simulate_fixed_tau(all_frames, tau=float(tau))
        drop = res["map_single"] - res["map_ours"]
        if drop < best_drop:
            best_drop, best_tau, best_res = drop, float(tau), res
    return best_tau, best_res


def simulate(all_frames, Etgt, kp=0.8, ki=0.10, beta=0.93,
             tau0=0.05, tau_lo=-0.5, tau_hi=1.0, integ_clip=3.0):
    """PI controller tracking energy target.

    integ_clip: anti-windup — clamps the integral term to prevent large
                overshoot/undershoot immediately after a step change in target.
    """
    n = len(all_frames)
    tau = tau0
    integ = 0.0
    E_ema = Etgt[0]
    prev_choice = None

    realized = np.empty(n)
    taus     = np.empty(n)
    is_super = np.zeros(n, bool)
    matches_ours   = []
    matches_single = []
    gt_count_all   = {}

    for i, (fr, first) in enumerate(all_frames):
        if first:
            prev_choice = None
            integ = 0.0   # reset integral at sequence boundary

        if prev_choice is None:
            choice = "base"
        else:
            pv = fr["ahat_super"] if prev_choice == "super" else fr["ahat_base"]
            choice = "super" if pv > tau else "base"
        prev_choice = choice

        E_frame = (E_SUPER if choice == "super" else E_BASE) + E_ROUTER
        realized[i] = E_frame
        taus[i] = tau
        is_super[i] = choice == "super"

        matches_ours.extend(fr[f"match_{choice}"])
        matches_single.extend(fr["match_super"])
        for c, k in fr["gt_count"].items():
            gt_count_all[c] = gt_count_all.get(c, 0) + k

        E_ema = beta * E_ema + (1 - beta) * E_frame
        e = Etgt[i] - E_ema
        integ = float(np.clip(integ + e, -integ_clip, integ_clip))  # anti-windup
        tau = float(np.clip(tau - kp * e - ki * integ, tau_lo, tau_hi))

    super_rate = is_super.sum() / n
    map_ours   = compute_map(matches_ours,   gt_count_all)
    map_single = compute_map(matches_single, gt_count_all)
    return {
        "realized":   realized,
        "taus":       taus,
        "is_super":   is_super,
        "super_rate": super_rate,
        "map_ours":   map_ours,
        "map_single": map_single,
        "matches_ours":   matches_ours,
        "matches_single": matches_single,
        "gt_count":       gt_count_all,
    }


RCPARAMS = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
}


def _load_frames(dump_path):
    data = pickle.load(open(dump_path, "rb"))
    seqs = data["seqs"]
    all_frames = []
    for seq_id, _, _ in SCENE_ORDER:
        if seq_id not in seqs:
            print(f"  [warn] {seq_id} not in dump, skipping")
            continue
        frames = seqs[seq_id]["frames"]
        for j, fr in enumerate(frames):
            all_frames.append((fr, j == 0))
    return all_frames


def rolling_map(matches_per_frame, gt_count_per_frame, win=100):
    """Compute rolling mAP@50:95 over a sliding window of frames."""
    n = len(matches_per_frame)
    maps = np.full(n, np.nan)
    for i in range(win - 1, n):
        m_win, g_win = [], {}
        for j in range(i - win + 1, i + 1):
            m_win.extend(matches_per_frame[j])
            for c, k in gt_count_per_frame[j].items():
                g_win[c] = g_win.get(c, 0) + k
        maps[i] = compute_map(m_win, g_win)
    return maps


# Paper-reported BDD100K numbers (from full val-set eval, not simulation subset)
MAP_SINGLE = 39.20
MAP_OURS   = 39.08


def _add_ap_inset(fig, ax, map_single=MAP_SINGLE, map_ours=MAP_OURS):
    """Small bar inset showing mAP@50:95 comparison."""
    # place inset at lower-left of ax
    ax_pos = ax.get_position()
    inset = fig.add_axes([ax_pos.x0 + 0.03,
                          ax_pos.y0 + 0.12,
                          0.13, 0.40])
    bars = inset.bar([0, 1], [map_single, map_ours],
                     color=["tab:blue", "tab:red"],
                     width=0.6, edgecolor="white", linewidth=0.5)
    # value labels on top of bars
    for bar, val in zip(bars, [map_single, map_ours]):
        inset.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                   f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    inset.set_xticks([0, 1])
    inset.set_xticklabels(["Single\ndet.", "Ours"], fontsize=7.5)
    inset.set_ylabel("mAP@50:95", fontsize=7.5)
    inset.set_ylim(map_ours - 2.0, map_single + 1.5)
    inset.tick_params(labelsize=7, length=2)
    inset.set_title("AP maintained", fontsize=7.5, pad=3)
    inset.spines[["top", "right"]].set_visible(False)


def _draw_perframe(ax, frames, tgt, realized_sm):
    """Per-frame energy tracking with two single-detector baselines."""
    ax.axhline(E_SUPER + E_ROUTER, color="tab:blue", lw=2.0, ls="-",
               label="YOLOv12m (single)", zorder=2)
    ax.axhline(E_BASE + E_ROUTER, color="tab:blue", lw=2.0, ls=":",
               label="YOLOv12s (single)", zorder=2)
    ax.plot(frames, tgt, color="black", ls="--", lw=2.0,
            label="Target budget", zorder=5)
    ax.plot(frames, realized_sm, color="tab:red", lw=2.0,
            label="Realized (ours)", zorder=4)
    ax.fill_between(frames, tgt, realized_sm, alpha=0.12, color="tab:red", zorder=3)
    ax.set_xlabel("Frame", fontsize=13)
    ax.set_ylabel("Energy (mJ)", fontsize=13)
    ax.set_xlim(frames[0], frames[-1])
    ax.grid(alpha=0.20, ls="--")
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper right", fontsize=10, frameon=True,
              framealpha=0.95, edgecolor="0.8", ncol=1)


def _draw_fps(ax, frames, tgt, realized_sm):
    """Per-frame FPS tracking with two single-detector baselines."""
    ax.axhline(FPS_SUPER, color="tab:blue", lw=2.0, ls="-",
               label="YOLOv12m (single)", zorder=2)
    ax.axhline(FPS_BASE, color="tab:blue", lw=2.0, ls=":",
               label="YOLOv12s (single)", zorder=2)
    ax.plot(frames, tgt, color="black", ls="--", lw=2.0,
            label="Target budget", zorder=5)
    ax.plot(frames, realized_sm, color="tab:red", lw=2.0,
            label="Realized (ours)", zorder=4)
    ax.fill_between(frames, tgt, realized_sm, alpha=0.12, color="tab:red", zorder=3)
    ax.set_xlabel("Frame", fontsize=13)
    ax.set_ylabel("FPS", fontsize=13)
    ax.set_xlim(frames[0], frames[-1])
    ax.grid(alpha=0.20, ls="--")
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper right", fontsize=10, frameon=True,
              framealpha=0.95, edgecolor="0.8", ncol=1)


def _draw_accumulated_with_ap(fig, axes, t_sec, realized, single_val,
                              matches_ours_per_frame, matches_single_per_frame,
                              gt_count_per_frame, map_ours, map_single,
                              roll_win=100):
    """Two-panel figure: (top) accumulated energy, (bottom) rolling mAP."""
    ax_e, ax_ap = axes
    dt = t_sec[1] - t_sec[0]
    acc_ours   = np.cumsum(realized)                         * dt * 1e-3
    acc_single = np.cumsum(np.full_like(realized, single_val)) * dt * 1e-3

    # ── top: accumulated energy ────────────────────────────────────────────────
    ax_e.plot(t_sec, acc_single, color="tab:blue", lw=2.2, label="Single detector")
    ax_e.plot(t_sec, acc_ours,   color="tab:red",  lw=2.0, label="Ours (adaptive)")
    ax_e.fill_between(t_sec, acc_single, acc_ours, alpha=0.15, color="tab:blue",
                      zorder=3)
    saved_pct = (acc_single[-1] - acc_ours[-1]) / acc_single[-1] * 100
    ax_e.annotate(f"−{saved_pct:.1f}% energy",
                  xy=(t_sec[-1], (acc_ours[-1] + acc_single[-1]) / 2),
                  xytext=(t_sec[-1] * 0.60, (acc_ours[-1] + acc_single[-1]) / 2 * 1.03),
                  fontsize=11, color="tab:blue", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.4))
    ax_e.set_ylabel("Accumulated energy (J)", fontsize=12)
    ax_e.set_xlim(t_sec[0], t_sec[-1])
    ax_e.grid(alpha=0.20, ls="--")
    ax_e.tick_params(labelsize=10)
    ax_e.legend(loc="upper left", fontsize=10, frameon=True,
                framealpha=0.95, edgecolor="0.8")

    # ── bottom: rolling mAP ────────────────────────────────────────────────────
    roll_ours   = rolling_map(matches_ours_per_frame,   gt_count_per_frame, roll_win)
    roll_single = rolling_map(matches_single_per_frame, gt_count_per_frame, roll_win)
    valid = ~np.isnan(roll_ours)
    ax_ap.plot(t_sec[valid], roll_single[valid] * 100, color="tab:blue", lw=2.2,
               label=f"Single detector (mAP={map_single*100:.1f})")
    ax_ap.plot(t_sec[valid], roll_ours[valid]   * 100, color="tab:red",  lw=2.0,
               label=f"Ours (mAP={map_ours*100:.1f})")
    ax_ap.set_xlabel("Time (s)", fontsize=12)
    ax_ap.set_ylabel("mAP@50:95 (%)", fontsize=12)
    ax_ap.set_xlim(t_sec[0], t_sec[-1])
    ax_ap.grid(alpha=0.20, ls="--")
    ax_ap.tick_params(labelsize=10)
    ax_ap.legend(loc="lower right", fontsize=10, frameon=True,
                 framealpha=0.95, edgecolor="0.8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="results/step3_eval/bdd100k/perframe_scenarios.pkl")
    ap.add_argument("--smooth_w", type=int, default=40)
    args = ap.parse_args()

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = _load_frames(args.dump)
    n = len(all_frames)
    frames = np.arange(n)

    plt.rcParams.update(RCPARAMS)

    # ── Figure 1: Target Energy budget tracking ──────────────────────────────────
    Etgt = energy_target(n)
    res_e = simulate(all_frames, Etgt)
    sm_e  = smooth(res_e["realized"], w=args.smooth_w)
    m_e   = compute_metrics(sm_e, Etgt, fps=10.0)
    print(f"[energy] MAE={m_e['mae']:.1f} mJ  SUPER={res_e['super_rate']*100:.1f}%")

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    _draw_perframe(ax, frames, Etgt, sm_e)
    fig.tight_layout(pad=0.5)
    out1 = str(out_dir / "fig_energy_budget_bdd.pdf")
    fig.savefig(out1, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[*] -> {out1}")

    # ── Figure 2: Target FPS budget tracking ────────────────────────────────────
    Ftgt = fps_target(n)
    # Convert FPS target → latency target → energy target using measured anchors
    L_tgt = 1000.0 / Ftgt - L_ROUTER
    # linearly interpolate energy from latency
    alpha = (np.clip(L_tgt, L_BASE, L_SUPER) - L_BASE) / (L_SUPER - L_BASE)
    Etgt_fps = E_BASE + alpha * (E_SUPER - E_BASE)
    res_f = simulate(all_frames, Etgt_fps)
    sm_E  = smooth(res_f["realized"], w=args.smooth_w)
    # convert realized energy back to FPS
    alpha_r = (sm_E - E_BASE) / (E_SUPER - E_BASE)
    sm_lat  = L_BASE + alpha_r * (L_SUPER - L_BASE)
    sm_fps  = 1000.0 / (sm_lat + L_ROUTER)
    m_f   = compute_metrics(sm_fps, Ftgt, fps=5.0, band=FPS_BASE - FPS_SUPER)
    print(f"[fps]    MAE={m_f['mae']:.2f} fps  SUPER={res_f['super_rate']*100:.1f}%")

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    _draw_fps(ax, frames, Ftgt, sm_fps)
    fig.tight_layout(pad=0.5)
    out2 = str(out_dir / "fig_fps_budget_bdd.pdf")
    fig.savefig(out2, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[*] -> {out2}")


if __name__ == "__main__":
    main()
