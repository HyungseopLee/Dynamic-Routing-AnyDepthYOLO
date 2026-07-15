"""Streaming (memory-safe) variant of online_budget_demo for the Jetson.

Identical Fig-8 scenario budget-tracking (night<->day / city<->highway / clear<->rainy
rows x {step, sawtooth} budgets), but two things differ from online_budget_demo:

  1. Frames are decoded LAZILY, one at a time, straight from the .mov -- never
     materialized into a per-family list. The original held every decoded BGR frame of
     all three families in RAM (~3000 x 720x1280x3 ~ 8 GB) and OOM-rebooted the 8 GB
     Orin. Here memory stays at one frame; each (family, budget) cell re-streams the
     videos (label counts give n / segment bounds up front, no decode needed).

  2. The PI controller uses a positional form with back-calculation anti-windup. The
     original conditional-integration scheme froze the integrator while saturated and
     got stuck at the all-SUPER rail on a falling budget; back-calculation un-saturates
     the instant the error reverses (validated on the KITTI tracking run).

Timed scope and everything else (live BASE/SUPER anchors, router-on-executed-path,
display smoothing) match online_budget_demo, whose render/schedule helpers are reused.

Step 1 — run live inference and save dump:

    python -m method_advantage_regress.jetson.online_budget_demo_stream \\
        --weight  finetuning_AnyDepthYOLO/weights/bdd100k/best.pt \\
        --policy  method_advantage_regress/outputs/bdd100k/policy_both_0.pt \\
        --scenarios method_advantage_regress/outputs/bdd100k/scenarios.json \\
        --mot_root  /media/data/bdd100k_mot/val \\
        --dump method_advantage_regress/outputs/bdd100k/online_budget_demo.json \\
        --out  method_advantage_regress/outputs/figures/fig_scenario_budget.pdf
        
# FPS   
PYTHONPATH=. python3 method_advantage_regress/jetson/online_budget_demo_stream.py \
    --base trt_bench/onnx/bdd_pooled/base.fp16.engine \
    --super trt_bench/onnx/bdd_pooled/super.fp16.engine \
    --router_engine trt_bench/onnx/bdd_pooled/router.fp16.engine \
    --policy method_advantage_regress/outputs/bdd100k/policy_both_0.pt \
    --mode fps \
    --kp 2.0 --ki 0.33 --beta 0.85 --warmup 60 --win 30 \
    --dump method_advantage_regress/outputs/bdd100k/jetson_fps_mode.json \
    --out  method_advantage_regress/outputs/bdd100k/jetson_fps_mode.pdf


# Energy        
method_advantage_regress/jetson/online_budget_demo_stream.py \
    --base trt_bench/onnx/bdd_pooled/base.fp16.engine \
    --super trt_bench/onnx/bdd_pooled/super.fp16.engine \
    --router_engine trt_bench/onnx/bdd_pooled/router.fp16.engine \
    --mode energy \
    --kp 2.0 --ki 0.33 --beta 0.85 --warmup 60 --win 30 \
    --dump method_advantage_regress/outputs/bdd100k/jetson_energy_mode.json \
    --out  method_advantage_regress/outputs/bdd100k/jetson_energy_mode.pdf

Step 2 — re-render from saved dump (instant, no inference):

    python -m method_advantage_regress.jetson.online_budget_demo_stream \\
        --dump method_advantage_regress/outputs/bdd100k/online_budget_demo.json \\
        --out  method_advantage_regress/outputs/figures/fig_scenario_budget.pdf \\
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

from method_advantage_regress.router.feature_tap import INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS  # noqa
from method_advantage_regress.eval.eval_video import (  # noqa
    parse_box_track, labeled_frames, load_policy, grid_vec)
from method_advantage_regress.jetson.online_budget_demo import (  # noqa
    OUT, cond_label, target_schedule, render)


def render_fps(dump, win, out):
    """Render FPS-mode budget tracking figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from method_advantage_regress.jetson.online_budget_demo import trail, SHORT, TITLES
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    fams = dump["fam_order"]
    fps_lo, fps_hi = dump["fps_lo"], dump["fps_hi"]
    ylo = fps_lo - 2.0; yhi = fps_hi + 2.0
    budgets = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]
    fig, axes = plt.subplots(len(fams), 2, figsize=(7.2, 1.9 * len(fams)), squeeze=False)
    print(f"\n{'scenario':<14}{'budget':<10}{'MAE(fps)':>10}{'SUPER%':>9}")
    for r, fam in enumerate(fams):
        bounds = dump["families"][fam]["bounds"]; labels = dump["families"][fam]["labels"]
        for c, (bkind, btitle) in enumerate(budgets):
            cell = dump["cells"][f"{fam}/{bkind}"]
            realized_lat = np.asarray(cell["realized"])
            fps_tgt = np.asarray(cell["target"])   # stored in fps
            n = len(realized_lat)
            sm_fps = 1000.0 / trail(realized_lat, win)
            mae = float(np.mean(np.abs(sm_fps - fps_tgt)))
            print(f"{fam:<14}{bkind:<10}{mae:>10.2f}{cell['super_rate']*100:>8.1f}%")
            ax = axes[r][c]
            for k in range(len(labels)):
                x0, x1 = bounds[k], bounds[k + 1]
                ax.axvspan(x0, x1, color="tab:blue" if k % 2 == 0 else "tab:orange",
                           alpha=0.07, zorder=0)
                ax.text((x0 + x1) / 2, fps_hi + 0.8,
                        SHORT.get(labels[k], labels[k]),
                        ha="center", va="bottom", fontsize=6.5, color="0.3")
                if k > 0:
                    ax.axvline(x0, color="0.6", ls="-", lw=0.6, alpha=0.5, zorder=1)
            ax.plot(fps_tgt, color="black", ls="--", lw=1.3, zorder=6)
            ax.plot(sm_fps, color="tab:red", lw=1.5, zorder=5)
            ax.set_ylim(ylo, yhi); ax.set_xlim(0, n)
            ax.grid(alpha=0.2, ls="--"); ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(btitle, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{TITLES.get(fam, fam)}\nFPS", fontsize=7.5)
            if r == len(fams) - 1:
                ax.set_xlabel("frame", fontsize=8)
            ax.text(0.02, 0.04, f"MAE={mae:.2f} fps", transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    handles = [Line2D([0], [0], color="black", ls="--", lw=1.3, label="target FPS $F^\\star(t)$"),
               Line2D([0], [0], color="tab:red", lw=1.5, label="realized mean FPS")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.03, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] -> {out}")


def render_energy(dump, win, out):
    """Render energy-mode budget tracking figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from method_advantage_regress.jetson.online_budget_demo import trail, SHORT, TITLES
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    fams = dump["fam_order"]
    e_base, e_super = dump["e_base"], dump["e_super"]
    e_lo = e_base + 0.10 * (e_super - e_base)
    e_hi = e_super - 0.10 * (e_super - e_base)
    ylo = e_base - 20.0; yhi = e_super + 40.0
    budgets = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]
    fig, axes = plt.subplots(len(fams), 2, figsize=(7.2, 1.9 * len(fams)), squeeze=False)
    print(f"\n{'scenario':<14}{'budget':<10}{'MAE(mJ)':>10}{'SUPER%':>9}")
    for r, fam in enumerate(fams):
        bounds = dump["families"][fam]["bounds"]; labels = dump["families"][fam]["labels"]
        for c, (bkind, btitle) in enumerate(budgets):
            cell = dump["cells"][f"{fam}/{bkind}"]
            realized_e = np.asarray(cell["realized"])   # stored as per-frame energy
            Etgt = np.asarray(cell["target"])
            n = len(realized_e)
            sm = trail(realized_e, win)
            mae = float(np.mean(np.abs(sm - Etgt)))
            print(f"{fam:<14}{bkind:<10}{mae:>10.1f}{cell['super_rate']*100:>8.1f}%")
            ax = axes[r][c]
            for k in range(len(labels)):
                x0, x1 = bounds[k], bounds[k + 1]
                ax.axvspan(x0, x1, color="tab:blue" if k % 2 == 0 else "tab:orange",
                           alpha=0.07, zorder=0)
                ax.text((x0 + x1) / 2, e_hi + 15.0,
                        SHORT.get(labels[k], labels[k]),
                        ha="center", va="bottom", fontsize=6.5, color="0.3")
                if k > 0:
                    ax.axvline(x0, color="0.6", ls="-", lw=0.6, alpha=0.5, zorder=1)
            ax.plot(Etgt, color="black", ls="--", lw=1.3, zorder=6)
            ax.plot(sm, color="tab:red", lw=1.5, zorder=5)
            ax.set_ylim(ylo, yhi); ax.set_xlim(0, n)
            ax.grid(alpha=0.2, ls="--"); ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(btitle, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{TITLES.get(fam, fam)}\nenergy (mJ)", fontsize=7.5)
            if r == len(fams) - 1:
                ax.set_xlabel("frame", fontsize=8)
            ax.text(0.02, 0.04, f"MAE={mae:.1f} mJ", transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    handles = [Line2D([0], [0], color="black", ls="--", lw=1.3, label="target energy $E^\\star(t)$"),
               Line2D([0], [0], color="tab:red", lw=1.5, label="realized mean energy")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.03, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default=None, help="AnyDepth .pt (eager backend)")
    ap.add_argument("--base", default=None, help="BASE TRT engine (.engine)")
    ap.add_argument("--super", default=None, help="SUPER TRT engine (.engine)")
    ap.add_argument("--router_engine", default=None, help="single TRT router engine (iv[,pv]->logit)")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--scenarios", default=str(OUT / "bdd100k/scenarios.json"))
    ap.add_argument("--mot_root", default="/media/data/bdd100k_mot/val")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--kp", type=float, default=0.020)
    ap.add_argument("--ki", type=float, default=0.004)
    ap.add_argument("--beta", type=float, default=0.85)
    ap.add_argument("--win", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT / "bdd100k/online_budget_demo.pdf"))
    ap.add_argument("--dump", default=str(OUT / "bdd100k/online_budget_demo.json"))
    ap.add_argument("--families", nargs="*", default=None, help="subset of scenario families to run")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--mode", default="latency", choices=["latency", "fps", "energy"],
                    help="control target: latency (ms), fps, or energy (mJ/frame)")
    args = ap.parse_args()

    if args.replot:
        dump = json.loads(Path(args.dump).read_text())
        mode = dump.get("mode", "latency")
        if mode == "fps":
            render_fps(dump, args.win, args.out)
        elif mode == "energy":
            render_energy(dump, args.win, args.out)
        else:
            render(dump, args.win, args.out)
        return

    dev = args.device if torch.cuda.is_available() else "cpu"

    if args.base and args.super:
        # TensorRT backend: BASE/SUPER (and optionally the router) run as TRT engines.
        # CUDA-event timed; correct for feat=both (computes iv AND pv from pooled taps).
        from method_advantage_regress.jetson.jetson_budget_track import TRTBackend
        from ultralytics.data.augment import LetterBox
        backend = TRTBackend(args.base, args.super, args.policy, args.grid, dev,
                             router_engine=args.router_engine)
        feat = backend.feat
        H = (args.imgsz[0] + 31) // 32 * 32
        W = (args.imgsz[1] + 31) // 32 * 32
        letterbox = LetterBox((H, W), auto=False, stride=32)
        print(f"[*] TRT backend: {args.base} / {args.super}"
              + (f" + router {args.router_engine}" if args.router_engine else " (eager router)")
              + f"  feat={feat}", flush=True)

        def run_frame(bgr, config):
            """Execute ONE path on TRT; return (measured_ms, ahat_for_next)."""
            img = letterbox(image=bgr)
            t = torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()).to(dev)
            x = (t.float() / 255.0).unsqueeze(0)
            ahat, lat = backend.run_frame(x, config)
            return lat, ahat
    else:
        # Eager PyTorch backend (kept for reference; only valid for feat=input policies).
        yolo = YOLO(args.weight, task="detect")
        yolo.model.to(dev).eval()
        N = getattr(yolo.model, "num_skippable_layers", 0)
        skip = {"super": [False] * N, "base": [True] * N}
        captured = {}
        for idx in STATE_LAYERS:
            yolo.model.model[idx].register_forward_hook(
                lambda m, i, o, k=idx: captured.__setitem__(k, o))
        net, ckpt, feat, is_gap = load_policy(args.policy, dev)
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

    scen = json.loads(Path(args.scenarios).read_text())
    label_dir = Path(args.mot_root) / "labels"
    video_dir = Path(args.mot_root) / "videos"

    def stream_family(order):
        """Yield (bgr, first_of_segment) for every labeled frame of a family, lazily."""
        for seg in order:
            gt = parse_box_track(label_dir / f"{seg['seq']}.json")
            first = True
            for _fidx, bgr in labeled_frames(video_dir / f"{seg['seq']}.mov", gt):
                yield bgr, first
                first = False

    # segment bounds / labels / frame counts straight from the label jsons (no decode)
    meta = {}
    for fam, order in scen.items():
        counts = [len(parse_box_track(label_dir / f"{seg['seq']}.json")) for seg in order]
        bounds = [0]
        for c in counts:
            bounds.append(bounds[-1] + c)
        meta[fam] = {"bounds": bounds, "labels": [cond_label(seg.get("cond", {})) for seg in order],
                     "n": bounds[-1], "order": order}
        print(f"[*] {fam}: {bounds[-1]} frames, {len(order)} segments", flush=True)

    # ---- warm-up: live BASE/SUPER anchors (device-level, from the first family) plus a
    # tau range probed across ALL families. The advantage distribution drifts by scene
    # (e.g. rainy frames genuinely favour SUPER), so a tau band calibrated on only the
    # first family is too narrow: the controller then saturates at a rail in other scenes
    # and cannot reach the operating point its budget needs (a scene-dependent tracking
    # bias). Spanning every family's Ahat range lets tau reach any scene's operating point.
    # For energy mode: read board power via INA3221 during warmup (sparse sampling)
    if args.mode == "energy":
        from method_advantage_regress.jetson.bench_trt_jetson import JetsonPower
        jp = JetsonPower()
        if not jp.available:
            print("[!] INA3221 not found; energy mode will use latency-proportional estimates")

    fam0 = next(iter(scen))
    wb, ws = [], []
    pb, ps = [], []   # power samples for energy mode
    wa_per_fam = {}   # per-family advantages for accurate tau initialisation
    wa_fam0 = []
    for i, (bgr, _first) in enumerate(stream_family(meta[fam0]["order"])):
        if i >= args.warmup:
            break
        lb, ab = run_frame(bgr, "base"); wb.append(lb); wa_fam0.append(ab)
        ls, as_ = run_frame(bgr, "super"); ws.append(ls); wa_fam0.append(as_)
        if args.mode == "energy" and i % 5 == 0:
            pb.append(jp.mw("VDD_IN") if jp.available else 0.0)
            ps.append(jp.mw("VDD_IN") if jp.available else 0.0)
    wa_per_fam[fam0] = wa_fam0
    wa = list(wa_fam0)   # global range still spans all families
    l_base = float(np.median(wb[len(wb) // 2:]))
    l_super = float(np.median(ws[len(ws) // 2:]))

    # Compute per-frame energy anchors (mJ) for energy mode
    # Power is ~constant per engine under locked clocks, so E = P_mean * L_mean
    if args.mode == "energy" and jp.available and pb:
        p_base_mw = float(np.mean(pb)); p_super_mw = float(np.mean(ps))
    else:
        # Fallback: use benchmark-measured ratio (SUPER draws ~1.70x BASE power on Orin)
        p_base_mw = 20700.0; p_super_mw = 20950.0  # mW, from bdd100k_720x1280.log
    e_base = p_base_mw * l_base / 1000.0   # mJ/frame
    e_super = p_super_mw * l_super / 1000.0

    if args.mode == "fps":
        fps_hi = 1000.0 / (l_base + 0.10 * (l_super - l_base))
        fps_lo = 1000.0 / (l_super - 0.10 * (l_super - l_base))
        lo = fps_lo; hi = fps_hi
    elif args.mode == "energy":
        lo = e_base + 0.10 * (e_super - e_base)
        hi = e_super - 0.10 * (e_super - e_base)
        print(f"[*] energy anchors: BASE={e_base:.1f}  SUPER={e_super:.1f} mJ  "
              f"band [{lo:.1f}, {hi:.1f}]", flush=True)
    else:
        lo = l_base + 0.10 * (l_super - l_base)
        hi = l_super - 0.10 * (l_super - l_base)
    # extend the Ahat probe over the remaining families (latency anchors stay from fam0)
    for fam in scen:
        if fam == fam0:
            continue
        wa_fam = []
        for i, (bgr, _first) in enumerate(stream_family(meta[fam]["order"])):
            if i >= args.warmup:
                break
            wa_fam.append(run_frame(bgr, "base")[1])
            wa_fam.append(run_frame(bgr, "super")[1])
        wa_per_fam[fam] = wa_fam
        wa.extend(wa_fam)
    TAU_HI = float(max(wa)) + 0.03
    TAU_LO = float(min(wa)) - 0.03
    tau0 = 0.5 * (TAU_HI + TAU_LO)
    # Auto-scale the PI gains to the actuator: tau lives in a band of width
    # (TAU_HI-TAU_LO) but the error is in ms over a span of ~(l_super-l_base). On BDD
    # the advantage range is tiny (~0.08) while latency errors are ~15 ms, so fixed
    # gains would slam tau between the rails (bang-bang). Scaling by the band ratio
    # makes "a full-scale latency error moves tau ~across its band", dataset-robust.
    # gscale normalises PI gains to the actuator band:
    # error is in the control domain (ms / fps / mJ), tau lives in [TAU_LO, TAU_HI].
    if args.mode == "energy":
        ctrl_span = max(e_super - e_base, 1e-6)  # mJ range
    else:
        ctrl_span = max(l_super - l_base, 1e-6)  # ms range (fps mode also controls in latency space)
    gscale = (TAU_HI - TAU_LO) / ctrl_span
    kp = args.kp * gscale; ki = args.ki * gscale
    print(f"[*] measured anchors: BASE={l_base:.2f}  SUPER={l_super:.2f} ms  "
          f"band [{lo:.2f}, {hi:.2f}] ({args.mode});  tau in [{TAU_LO:.3f}, {TAU_HI:.3f}]  "
          f"gscale={gscale:.4f} -> kp={kp:.2e} ki={ki:.2e}", flush=True)

    # Pre-convert fps target array to latency for fps mode (control always in latency/energy space)
    def to_ctrl(tgt_arr):
        """Convert target schedule to the internal control domain."""
        if args.mode == "fps":
            return 1000.0 / tgt_arr   # fps → ms (note: high fps = low latency)
        return tgt_arr  # latency (ms) or energy (mJ) already in control domain

    def online_loop(order, tgt, fam):
        """Run PI controller. tgt is in display units (ms/fps/mJ); control is in latency or energy."""
        n = len(tgt)
        ctrl = to_ctrl(tgt)   # internal control signal array
        wa_fam = wa_per_fam.get(fam, wa)
        # Convert first target to latency fraction for tau initialisation
        if args.mode == "energy":
            l_tgt0 = l_base + (ctrl[0] - e_base) / max(e_super - e_base, 1e-6) * (l_super - l_base)
        else:
            l_tgt0 = ctrl[0]  # already in ms for latency and fps modes
        init_pct = float(np.clip((l_tgt0 - l_base) / (l_super - l_base), 0.0, 1.0))
        tau_init = float(np.clip(np.quantile(wa_fam, 1.0 - init_pct), TAU_LO, TAU_HI))
        config = "base"; tau = tau_init; integ = 0.0
        sig_ema = float(ctrl[0])
        realized = np.full(n, np.nan); n_super = 0; produced = 0
        for t, (bgr, first) in enumerate(stream_family(order)):
            if first:
                config = "base"
            lat, ahat = run_frame(bgr, config)
            produced = t + 1; n_super += (config == "super")
            # Realized signal in the control domain
            if args.mode == "energy":
                sig_realized = e_base if config == "base" else e_super
                realized[t] = sig_realized
            else:
                sig_realized = lat   # latency (ms) for both latency and fps modes
                realized[t] = lat
            sig_ema = args.beta * sig_ema + (1 - args.beta) * sig_realized
            # Flush integrator on sudden budget tightening (ctrl drops in latency/energy = budget tightens)
            if t > 0 and (ctrl[t] - ctrl[t - 1]) < -1.0:
                integ = 0.0
            e = ctrl[t] - sig_ema
            integ += e
            tau_un = tau0 - kp * e - ki * integ
            tau = float(np.clip(tau_un, TAU_LO, TAU_HI))
            if ki:
                integ += (tau_un - tau) / ki
            config = "super" if ahat > tau else "base"
        return realized[:produced], n_super / max(produced, 1)

    fams = [f for f in scen.keys() if not args.families or f in args.families]
    results = {}; families_meta = {}
    print(f"\n{'scenario':<14}{'budget':<10}{'SUPER%':>9}", flush=True)
    for fam in fams:
        families_meta[fam] = {"bounds": meta[fam]["bounds"], "labels": meta[fam]["labels"]}
        n = meta[fam]["n"]
        for bkind in ("step", "sawtooth"):
            tgt = target_schedule(bkind, n, lo, hi)
            t_cell = time.perf_counter()
            realized, super_rate = online_loop(meta[fam]["order"], tgt, fam)
            tgt = tgt[:len(realized)]
            print(f"    [{fam}/{bkind}] {len(realized)} frames in {time.perf_counter()-t_cell:.1f}s",
                  flush=True)
            results[f"{fam}/{bkind}"] = {"super_rate": super_rate,
                                         "realized": realized.tolist(),
                                         "target": tgt.tolist()}
            print(f"{fam:<14}{bkind:<10}{super_rate*100:>8.1f}%", flush=True)

    dump = {"l_base": l_base, "l_super": l_super, "win": args.win,
            "kp": args.kp, "ki": args.ki, "mode": args.mode,
            "fam_order": fams, "families": families_meta, "cells": results}
    if args.mode == "fps":
        dump["fps_lo"] = fps_lo; dump["fps_hi"] = fps_hi
    elif args.mode == "energy":
        dump["e_base"] = e_base; dump["e_super"] = e_super
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dump, open(args.dump, "w"))
    print(f"[*] -> {args.dump}", flush=True)
    if args.mode == "fps":
        render_fps(dump, args.win, args.out)
    elif args.mode == "energy":
        render_energy(dump, args.win, args.out)
    else:
        render(dump, args.win, args.out)


if __name__ == "__main__":
    main()
