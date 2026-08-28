"""On-device ONLINE latency-budget demo -- live version of the scenario budget-tracking
figure. Runs the full closed loop LIVE: each frame executes EXACTLY ONE detector path,
its latency is MEASURED on the fly, the router predicts the next-frame advantage from
the features of the executed path, and a PI controller adjusts tau so the realized
latency tracks a time-varying budget L*(t). Nothing is replayed.

Rows = scenario families (night<->daytime, city<->highway, clear<->rainy),
columns = {step, sawtooth} budgets. Self-calibrating warm-up measures live
BASE/SUPER latency anchors on this device.

NOTE: loads all frames into RAM upfront (~8 GB for 3000 frames). For memory-
constrained devices (e.g. Jetson) use online_budget_demo_stream.py instead.

Step 1 — run live inference and save dump:

    python -m step4_deploy.online_budget_demo \
        --weight  results/step1_finetune/weights/bdd100k/best.pt \
        --router  results/step2_router/weights/bdd100k/router_g2x2_both_s0.pt \
        --scenarios results/step3_eval/bdd100k/scenarios.json \
        --mot_root  /media/data/bdd100k_mot/val \
        --dump results/step4_deploy/control/online_budget_demo.json \
        --out  results/figures/fig_scenario_budget.pdf

Step 2 — re-render from saved dump (instant, no inference):
    python -m step4_deploy.online_budget_demo \
        --dump results/step4_deploy/control/online_budget_demo.json \
        --out  results/figures/fig_scenario_budget.pdf \
        --replot [--win 60]

The printed MAE is computed on the smoothed (centered moving-average, --win frames)
realized latency vs the target budget. Raw per-frame MAE is ~3 ms; smoothed MAE
reflects tracking quality of the visible curve.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ultralytics import YOLO  # noqa

from router.feature_tap import INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS  # noqa
from step3_eval.eval_video import (  # noqa
    parse_box_track, labeled_frames, load_router, grid_vec)

OUT = Path(__file__).resolve().parents[1] / "results/step3_eval"
TITLES = {"night_dawn": "night $\\leftrightarrow$ daytime",
          "city_highway": "city $\\leftrightarrow$ highway",
          "clear_rainy": "clear $\\leftrightarrow$ rainy"}
SHORT = {"night": "night", "daytime": "day", "dawn/dusk": "dawn",
         "city street": "city", "highway": "hwy", "clear": "clear", "rainy": "rain"}


def cond_label(cond):
    if isinstance(cond, dict):
        for k in ("timeofday", "scene", "weather"):
            if cond.get(k):
                return cond[k]
        return ""
    return cond or ""


def target_schedule(kind, n, lo, hi):
    t = np.arange(n)
    if kind == "step":
        # Fewer, longer holds than the 5 condition segments, so each constant-budget
        # hold spans a night<->day (condition) change rather than aligning with one.
        seg = np.array([0.30, 0.80, 0.50])
        edges = np.linspace(0, n, len(seg) + 1).astype(int)
        L = np.empty(n)
        for i, frac in enumerate(seg):
            L[edges[i]:edges[i + 1]] = lo + frac * (hi - lo)
        return L
    if kind == "sawtooth":
        period = max(1, n // 4)
        return lo + ((t % period) / period) * (hi - lo)
    raise ValueError(kind)


def trail(x, w):
    """Zero-phase (centered) moving average. This figure is an offline visualization
    of whether the realized latency tracked the budget, not a real-time readout, so we
    use a symmetric window: it has no phase lag, exposing the controller's true tracking
    quality rather than the ~w/2-frame delay a causal trailing average would introduce.
    The underlying closed loop is unchanged; only the display smoothing is centered."""
    x = np.asarray(x, float)
    h = w // 2
    xp = np.pad(x, (h, w - 1 - h), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def render(dump, win, out):
    """Build the scenario x budget grid from a saved dump, smoothing the realized
    latency with a trailing window of `win` frames (re-smoothing is instant)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    l_base, l_super = dump["l_base"], dump["l_super"]
    fams = dump["fam_order"]
    budgets = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]
    lo = l_base + 0.10 * (l_super - l_base); hi = l_super - 0.10 * (l_super - l_base)
    ylo, yhi = l_base - 1.2, l_super + 1.6
    fig, axes = plt.subplots(len(fams), 2, figsize=(7.2, 1.9 * len(fams)), squeeze=False)
    print(f"\n{'scenario':<14}{'budget':<10}{'MAE(ms)':>9}{'SUPER%':>9}")
    for r, fam in enumerate(fams):
        bounds = dump["families"][fam]["bounds"]; labels = dump["families"][fam]["labels"]
        for c, (bkind, btitle) in enumerate(budgets):
            cell = dump["cells"][f"{fam}/{bkind}"]
            realized = np.asarray(cell["realized"]); Ltgt = np.asarray(cell["target"])
            n = len(realized)
            sm = trail(realized, win)
            mae = float(np.mean(np.abs(sm - Ltgt)))
            print(f"{fam:<14}{bkind:<10}{mae:>9.2f}{cell['super_rate']*100:>8.1f}%")
            ax = axes[r][c]
            for k in range(len(labels)):
                x0, x1 = bounds[k], bounds[k + 1]
                ax.axvspan(x0, x1, color="tab:blue" if k % 2 == 0 else "tab:orange",
                           alpha=0.07, zorder=0)
                ax.text((x0 + x1) / 2, hi + 0.5, SHORT.get(labels[k], labels[k]),
                        ha="center", va="bottom", fontsize=6.5, color="0.3")
                if k > 0:
                    ax.axvline(x0, color="0.6", ls="-", lw=0.6, alpha=0.5, zorder=1)
            ax.plot(Ltgt, color="black", ls="--", lw=1.3, zorder=6)
            ax.plot(sm, color="tab:red", lw=1.5, zorder=5)
            ax.set_ylim(ylo, yhi); ax.set_xlim(0, n)
            ax.grid(alpha=0.2, ls="--"); ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(btitle, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{TITLES.get(fam, fam)}\nlatency (ms)", fontsize=7.5)
            if r == len(fams) - 1:
                ax.set_xlabel("frame", fontsize=8)
            ax.text(0.02, 0.04, f"MAE={mae:.2f} ms", transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    handles = [Line2D([0], [0], color="black", ls="--", lw=1.3, label="target budget $L^\\star(t)$"),
               Line2D([0], [0], color="tab:red", lw=1.5,
                      label="realized mean latency")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.03, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] -> {out}")


def replot(args):
    dump = json.loads(Path(args.dump).read_text())
    render(dump, args.win, args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--scenarios", default=str(OUT / "bdd100k/scenarios.json"))
    ap.add_argument("--mot_root", default="/media/data/bdd100k_mot/val")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fps", type=float, default=0.0, help="real-time pacing fps; 0 = as fast as possible")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--kp", type=float, default=0.020)
    ap.add_argument("--ki", type=float, default=0.004)
    ap.add_argument("--beta", type=float, default=0.85)
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT / "bdd100k/online_budget_demo.pdf"))
    ap.add_argument("--dump", default=str(OUT / "bdd100k/online_budget_demo.json"))
    ap.add_argument("--replot", action="store_true",
                    help="re-render from the saved dump (re-smooth with --win); no inference")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    if args.replot:
        replot(args); return

    yolo = YOLO(args.weight, task="detect")
    yolo.model.to(dev).eval()
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip = {"super": [False] * N, "base": [True] * N}
    captured = {}
    for idx in STATE_LAYERS:
        yolo.model.model[idx].register_forward_hook(
            lambda m, i, o, k=idx: captured.__setitem__(k, o))
    net, ckpt, feat, is_gap = load_router(args.router, dev)
    pid = {"super": torch.ones(1, dtype=torch.long, device=dev),
           "base": torch.zeros(1, dtype=torch.long, device=dev)}

    def cuda_sync():
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    def run_frame(bgr, config):
        """Execute ONE path; return (measured_ms, ahat_for_next)."""
        captured.clear()
        r = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                         iou=0.7, skip=skip[config], verbose=False, device=dev)[0]
        det_ms = r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]
        xi = grid_vec(captured, INPUT_LEVEL_LAYERS, args.grid).unsqueeze(0)
        xp = grid_vec(captured, PRED_LEVEL_LAYERS, args.grid).unsqueeze(0) if feat != "input" else None
        if is_gap:
            xi = xi.mean(dim=(2, 3))
            xp = xp.mean(dim=(2, 3)) if xp is not None else None
        cuda_sync(); t0 = time.perf_counter()
        with torch.no_grad():
            ahat = float(net.logit(xi, xp, pid[config]))
        cuda_sync(); rtr_ms = (time.perf_counter() - t0) * 1000.0
        return det_ms + rtr_ms, ahat

    # ---- build the streams for each scenario family (concatenate sequences) ----
    scen = json.loads(Path(args.scenarios).read_text())
    label_dir = Path(args.mot_root) / "labels"
    video_dir = Path(args.mot_root) / "videos"
    streams = {}            # fam -> (frames[list of (bgr, first)], bounds, labels)
    for fam, order in scen.items():
        frames, bounds, labels = [], [], []
        for seg in order:
            bounds.append(len(frames)); labels.append(cond_label(seg.get("cond", {})))
            gt = parse_box_track(label_dir / f"{seg['seq']}.json")
            first = True
            for fidx, bgr in labeled_frames(video_dir / f"{seg['seq']}.mov", gt):
                frames.append((bgr, first)); first = False
        bounds.append(len(frames))
        streams[fam] = (frames, bounds, labels)
        print(f"[*] {fam}: {len(frames)} frames, {len(order)} segments")

    # ---- warm-up on the first family's frames: live BASE/SUPER latency anchors and
    #      the router's advantage range (to set self-calibrating tau bounds) ----
    f0 = streams[next(iter(streams))][0]
    wb, ws, wa = [], [], []
    for i in range(min(args.warmup, len(f0))):
        lb, ab = run_frame(f0[i][0], "base"); wb.append(lb); wa.append(ab)
        ls, as_ = run_frame(f0[i][0], "super"); ws.append(ls); wa.append(as_)
    l_base = float(np.median(wb[len(wb)//2:]))
    l_super = float(np.median(ws[len(ws)//2:]))
    lo = l_base + 0.10 * (l_super - l_base)
    hi = l_super - 0.10 * (l_super - l_base)
    # tau bounds bracket the observed advantage range: tau>=TAU_HI -> all base,
    # tau<=TAU_LO -> all super, so the controller can always reach either extreme.
    TAU_HI = float(max(wa)) + 0.03
    TAU_LO = float(min(wa)) - 0.03
    print(f"[*] measured anchors: BASE={l_base:.2f}  SUPER={l_super:.2f} ms  "
          f"band [{lo:.2f}, {hi:.2f}];  tau in [{TAU_LO:.3f}, {TAU_HI:.3f}]")

    def online_loop(frames, Ltgt):
        n = len(frames)
        # start at the upper tau clip (all-base, matching the initial config) and seed
        # the EMA with the base anchor, so there is no spurious start-up error.
        config = "base"; tau = TAU_HI; integ = 0.0
        L_ema = l_base
        realized = np.empty(n); n_super = 0
        period = 1.0 / args.fps if args.fps > 0 else 0.0
        for t, (bgr, first) in enumerate(frames):
            if first:
                config = "base"
            wall0 = time.perf_counter()
            lat, ahat = run_frame(bgr, config)
            realized[t] = lat; n_super += (config == "super")
            L_ema = args.beta * L_ema + (1 - args.beta) * lat
            e = Ltgt[t] - L_ema
            # PI with conditional-integration anti-windup: only accumulate the
            # integral when the resulting tau is not clipped (prevents the start-up
            # windup that can otherwise trap the loop at one extreme).
            tau_un = tau - args.kp * e - args.ki * (integ + e)
            tau_cl = float(np.clip(tau_un, TAU_LO, TAU_HI))
            if tau_un == tau_cl:
                integ += e
            tau = tau_cl
            config = "super" if ahat > tau else "base"
            if period:
                dt = period - (time.perf_counter() - wall0)
                if dt > 0:
                    time.sleep(dt)
        return realized, n_super / n

    # ---- run the full grid LIVE ----
    fams = list(streams.keys())
    budgets = ["step", "sawtooth"]
    results = {}
    families_meta = {}
    print(f"\n{'scenario':<14}{'budget':<10}{'SUPER%':>9}")
    for fam in fams:
        frames, bounds, labels = streams[fam]
        families_meta[fam] = {"bounds": bounds, "labels": labels}
        n = len(frames)
        for bkind in budgets:
            Ltgt = target_schedule(bkind, n, lo, hi)
            realized, super_rate = online_loop(frames, Ltgt)
            results[f"{fam}/{bkind}"] = {"super_rate": super_rate,
                                         "realized": realized.tolist(),
                                         "target": Ltgt.tolist()}
            print(f"{fam:<14}{bkind:<10}{super_rate*100:>8.1f}%", flush=True)

    dump = {"l_base": l_base, "l_super": l_super, "win": args.win,
            "kp": args.kp, "ki": args.ki, "fps": args.fps,
            "fam_order": fams, "families": families_meta, "cells": results}
    json.dump(dump, open(args.dump, "w"))
    print(f"[*] -> {args.dump}")
    render(dump, args.win, args.out)


if __name__ == "__main__":
    main()
