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

Step 1a — TRT backend (base + super + router engines, RTX 3090):

    python -m method_advantage_regress.jetson.online_budget_demo_stream \
        --base   method_advantage_regress/jetson/onnx/bdd_pooled/base.fp16.engine \
        --super  method_advantage_regress/jetson/onnx/bdd_pooled/super.fp16.engine \
        --router method_advantage_regress/outputs/bdd100k/router_both_0.pt \
        --router_engine method_advantage_regress/jetson/onnx/bdd_pooled/router.fp16.engine \
        --scenarios method_advantage_regress/outputs/bdd100k/scenarios.json \
        --mot_root  /media/data/bdd100k_mot/val \
        --kp 1.0 --ki 0.10 --beta 0.75 --warmup 60 --win 30 \
        --mode fps
        # output auto-named: online_budget_demo_trt_<engdir>_<HxW>_<mode>_b<beta>_kp<kp>_ki<ki>_win<win>.{json,pdf}
        # --mode: latency (default, ms) | fps | energy (mJ/frame via NVML, needs pynvml)
        # final FPS-tracking figure params: --kp 1.0 --ki 0.10 --beta 0.75 --win 30

Step 1b — PyTorch eager backend:

    python -m method_advantage_regress.jetson.online_budget_demo_stream \\
        --weight  finetuning_AnyDepthYOLO/weights/bdd100k/best.pt \\
        --router  method_advantage_regress/outputs/bdd100k/router_both_0.pt \\
        --scenarios method_advantage_regress/outputs/bdd100k/scenarios.json \\
        --mot_root  /media/data/bdd100k_mot/val \\
        --kp 0.020 --ki 0.004 --beta 0.85 --warmup 30 --win 60
        # output auto-named: online_budget_demo_torch_b0.85_kp0.02_ki0.004_warmup30_window60.{json,pdf}

Step 2 — re-render from saved dump (instant, no inference):

    python -m method_advantage_regress.jetson.online_budget_demo_stream \\
        --dump method_advantage_regress/outputs/bdd100k/online_budget_demo_trt_b0.93_kp0.28_ki0.06_warmup60_window60.json \\
        --replot [--win 60]
        # --out can be specified to override the default PDF path

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
    parse_box_track, labeled_frames, load_router, grid_vec)
from method_advantage_regress.jetson.online_budget_demo import (  # noqa
    OUT, SHORT, TITLES, cond_label, target_schedule, render)


def render_fps(dump, win, out):
    """Like render() but plots realized/target in FPS space instead of ms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from method_advantage_regress.jetson.online_budget_demo import trail
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    l_base, l_super = dump["l_base"], dump["l_super"]
    fps_base = 1000.0 / l_base
    fps_super = 1000.0 / l_super
    fams = dump["fam_order"]
    budgets = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]
    ylo, yhi = fps_super - 2.0, fps_base + 2.0
    fig, axes = plt.subplots(len(fams), 2, figsize=(7.2, 1.9 * len(fams)), squeeze=False)
    print(f"\n{'scenario':<14}{'budget':<10}{'MAE(fps)':>10}{'SUPER%':>9}")
    for r, fam in enumerate(fams):
        bounds = dump["families"][fam]["bounds"]
        labels = dump["families"][fam]["labels"]
        for c, (bkind, btitle) in enumerate(budgets):
            cell = dump["cells"][f"{fam}/{bkind}"]
            # realized and target stored in latency (ms); convert to FPS for display
            realized_lat = np.asarray(cell["realized"])
            Ltgt = np.asarray(cell["target"])
            fps_tgt = 1000.0 / Ltgt
            n = len(realized_lat)
            # window FPS = frames / total time = 1000/mean(lat) (harmonic mean of
            # per-frame fps); arithmetic mean of 1000/lat is biased high (Jensen)
            sm = 1000.0 / trail(realized_lat, win)
            mae = float(np.mean(np.abs(sm - fps_tgt)))
            print(f"{fam:<14}{bkind:<10}{mae:>10.2f}{cell['super_rate']*100:>8.1f}%")
            ax = axes[r][c]
            for k in range(len(labels)):
                x0, x1 = bounds[k], bounds[k + 1]
                ax.axvspan(x0, x1, color="tab:blue" if k % 2 == 0 else "tab:orange",
                           alpha=0.07, zorder=0)
                ax.text((x0 + x1) / 2, yhi - 0.5, SHORT.get(labels[k], labels[k]),
                        ha="center", va="top", fontsize=6.5, color="0.3")
                if k > 0:
                    ax.axvline(x0, color="0.6", ls="-", lw=0.6, alpha=0.5, zorder=1)
            ax.axhline(fps_base, color="tab:blue", lw=1.0, ls=":", alpha=0.6,
                       label=f"BASE only ({fps_base:.1f} fps)")
            ax.axhline(fps_super, color="tab:blue", lw=1.0, ls="-", alpha=0.6,
                       label=f"SUPER only ({fps_super:.1f} fps)")
            ax.plot(fps_tgt, color="black", ls="--", lw=1.3, zorder=6)
            ax.plot(sm, color="tab:red", lw=1.5, zorder=5)
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
    """Like render() but in energy space (mJ/frame); anchors/realized/target are mJ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from method_advantage_regress.jetson.online_budget_demo import trail
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    e_base, e_super = dump["l_base"], dump["l_super"]   # mJ anchors (generic keys)
    fams = dump["fam_order"]
    budgets = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]
    span = e_super - e_base
    ylo, yhi = e_base - 0.1 * span, e_super + 0.1 * span
    fig, axes = plt.subplots(len(fams), 2, figsize=(7.2, 1.9 * len(fams)), squeeze=False)
    print(f"\n{'scenario':<14}{'budget':<10}{'MAE(mJ)':>10}{'SUPER%':>9}")
    for r, fam in enumerate(fams):
        bounds = dump["families"][fam]["bounds"]
        labels = dump["families"][fam]["labels"]
        for c, (bkind, btitle) in enumerate(budgets):
            cell = dump["cells"][f"{fam}/{bkind}"]
            realized = np.asarray(cell["realized"])
            tgt = np.asarray(cell["target"])
            n = len(realized)
            sm = trail(realized, win)
            mae = float(np.mean(np.abs(sm - tgt)))
            print(f"{fam:<14}{bkind:<10}{mae:>10.2f}{cell['super_rate']*100:>8.1f}%")
            ax = axes[r][c]
            for k in range(len(labels)):
                x0, x1 = bounds[k], bounds[k + 1]
                ax.axvspan(x0, x1, color="tab:blue" if k % 2 == 0 else "tab:orange",
                           alpha=0.07, zorder=0)
                ax.text((x0 + x1) / 2, yhi - 0.02 * span, SHORT.get(labels[k], labels[k]),
                        ha="center", va="top", fontsize=6.5, color="0.3")
                if k > 0:
                    ax.axvline(x0, color="0.6", ls="-", lw=0.6, alpha=0.5, zorder=1)
            ax.axhline(e_base, color="tab:blue", lw=1.0, ls=":", alpha=0.6,
                       label=f"BASE only ({e_base:.0f} mJ)")
            ax.axhline(e_super, color="tab:blue", lw=1.0, ls="-", alpha=0.6,
                       label=f"SUPER only ({e_super:.0f} mJ)")
            ax.plot(tgt, color="black", ls="--", lw=1.3, zorder=6)
            ax.plot(sm, color="tab:red", lw=1.5, zorder=5)
            ax.set_ylim(ylo, yhi); ax.set_xlim(0, n)
            ax.grid(alpha=0.2, ls="--"); ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(btitle, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{TITLES.get(fam, fam)}\nEnergy (mJ/frame)", fontsize=7.5)
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
    ap.add_argument("--router", default=None)
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
    ap.add_argument("--out", default=None, help="output PDF path (auto-named from hyperparams if omitted)")
    ap.add_argument("--dump", default=None, help="dump JSON path (auto-named from hyperparams if omitted)")
    ap.add_argument("--families", nargs="*", default=None, help="subset of scenario families to run")
    ap.add_argument("--mode", default="latency", choices=["latency", "fps", "energy"],
                    help="tracking target space: latency (ms), fps, or energy (mJ/frame)")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    # Auto-name outputs so every run is fully traceable
    if args.base and args.super:
        # e.g. "bdd_pooled" from the engine directory name
        eng_name = Path(args.base).parent.name
        backend_tag = f"trt_{eng_name}"
    else:
        backend_tag = "torch"
    res_tag = f"{args.imgsz[0]}x{args.imgsz[1]}"
    hp_tag = f"b{args.beta}_kp{args.kp}_ki{args.ki}_win{args.win}"
    stem = f"online_budget_demo_{backend_tag}_{res_tag}_{args.mode}_{hp_tag}"
    if args.out is None:
        args.out = str(OUT / f"bdd100k/{stem}.pdf")
    if args.dump is None:
        args.dump = str(OUT / f"bdd100k/{stem}.json")

    if args.replot:
        dump = json.loads(Path(args.dump).read_text())
        mode = dump.get("mode", "latency")
        {"fps": render_fps, "energy": render_energy}.get(mode, render)(dump, args.win, args.out)
        return

    dev = args.device if torch.cuda.is_available() else "cpu"

    if args.base and args.super:
        # TensorRT backend: BASE/SUPER (and optionally the router) run as TRT engines.
        # CUDA-event timed; correct for feat=both (computes iv AND pv from pooled taps).
        from method_advantage_regress.jetson.jetson_budget_track import TRTBackend
        from ultralytics.data.augment import LetterBox
        backend = TRTBackend(args.base, args.super, args.router, args.grid, dev,
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
                lambda _m, _i, o, k=idx: captured.__setitem__(k, o))
        net, _ckpt, feat, is_gap = load_router(args.router, dev)
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

    # Energy mode: per-frame cost = GPU power (NVML) x latency, in mJ. Energy/frame is
    # linear in the SUPER rate (like latency, unlike FPS), so the whole latency-mode
    # pipeline (anchors, feedforward, PI) is reused with mJ anchors instead of ms.
    if args.mode == "energy":
        import pynvml
        pynvml.nvmlInit()
        gpu_idx = int(dev.split(":")[1]) if ":" in dev else 0
        nvml_h = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

        def frame_cost(lat):
            pw = pynvml.nvmlDeviceGetPowerUsage(nvml_h) / 1000.0   # mW -> W
            return pw * lat                                        # W * ms = mJ
    else:
        def frame_cost(lat):
            return lat

    _run_frame_raw = run_frame

    def run_frame(bgr, config):
        lat, ahat = _run_frame_raw(bgr, config)
        return frame_cost(lat), ahat

    scen = json.loads(Path(args.scenarios).read_text())
    label_dir = Path(args.mot_root) / "labels"
    video_dir = Path(args.mot_root) / "videos"

    def stream_family(order):
        """Yield (bgr, first_of_segment) for every labeled frame of a family, lazily."""
        for seg in order:
            gt = parse_box_track(label_dir / f"{seg['seq']}.json")
            first = True
            for _, bgr in labeled_frames(video_dir / f"{seg['seq']}.mov", gt):
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
    fam0 = next(iter(scen))
    wb, ws = [], []
    wa_per_fam = {}   # per-family advantages for accurate tau initialisation
    wa_fam0 = []
    for i, (bgr, _) in enumerate(stream_family(meta[fam0]["order"])):
        if i >= args.warmup:
            break
        lb, ab = run_frame(bgr, "base"); wb.append(lb); wa_fam0.append(ab)
        ls, as_ = run_frame(bgr, "super"); ws.append(ls); wa_fam0.append(as_)
    wa_per_fam[fam0] = wa_fam0
    wa = list(wa_fam0)   # global range still spans all families
    l_base = float(np.median(wb[len(wb) // 2:]))
    l_super = float(np.median(ws[len(ws) // 2:]))
    lo = l_base + 0.10 * (l_super - l_base)
    hi = l_super - 0.10 * (l_super - l_base)
    # FPS mode: target is defined in FPS space, then converted to latency for the PI controller.
    # (per-frame, only BASE or SUPER is achievable; the average latency tracks the FPS target.)
    if args.mode == "fps":
        fps_hi = 1000.0 / lo   # higher FPS corresponds to lower latency (more BASE)
        fps_lo = 1000.0 / hi   # lower FPS corresponds to higher latency (more SUPER)
    # extend the Ahat probe over the remaining families (latency anchors stay from fam0)
    for fam in scen:
        if fam == fam0:
            continue
        wa_fam = []
        for i, (bgr, _) in enumerate(stream_family(meta[fam]["order"])):
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
    fps_base  = 1000.0 / l_base
    fps_super = 1000.0 / l_super
    if args.mode == "fps":
        # PI error is in FPS units → scale gscale by FPS span instead of latency span
        gscale = (TAU_HI - TAU_LO) / max(fps_base - fps_super, 1e-6)
    else:
        gscale = (TAU_HI - TAU_LO) / max(l_super - l_base, 1e-6)
    kp = args.kp * gscale; ki = args.ki * gscale
    print(f"[*] measured anchors: BASE={l_base:.2f}ms ({fps_base:.1f}fps)  "
          f"SUPER={l_super:.2f}ms ({fps_super:.1f}fps)  "
          f"gscale={gscale:.4f} -> kp={kp:.2e} ki={ki:.2e}", flush=True)

    def online_loop(order, Ltgt, fam):
        """Ltgt is always in latency (ms); in fps mode, EMA and error are in FPS space."""
        n = len(Ltgt)
        wa_fam = wa_per_fam.get(fam, wa)
        init_pct = float(np.clip((Ltgt[0] - l_base) / (l_super - l_base), 0.0, 1.0))
        tau_init = float(np.clip(np.quantile(wa_fam, 1.0 - init_pct), TAU_LO, TAU_HI))
        config = "base"; tau = tau_init; integ = 0.0
        # Feedforward: the target is known, so compute the open-loop tau it needs
        # per-frame (mean latency is linear in SUPER rate p, and p maps to the
        # (1-p)-quantile of the advantage distribution). PI only corrects the residual,
        # which removes the ramp-tracking lag a pure PI has on sawtooth targets.
        pct = np.clip((np.asarray(Ltgt) - l_base) / max(l_super - l_base, 1e-6), 0.0, 1.0)
        tau_ff = np.clip(np.quantile(wa_fam, 1.0 - pct), TAU_LO, TAU_HI)
        # In fps mode: EMA tracks FPS directly (avoids 1/x nonlinearity amplifying error)
        fps_tgt0 = 1000.0 / Ltgt[0]
        fps_ema = fps_tgt0 if args.mode == "fps" else None
        L_ema   = float(Ltgt[0])
        realized = np.full(n, np.nan); n_super = 0; produced = 0
        for t, (bgr, first) in enumerate(stream_family(order)):
            if first:
                config = "base"
            lat, ahat = run_frame(bgr, config)
            realized[t] = lat; produced = t + 1; n_super += (config == "super")

            if args.mode == "fps":
                # EMA the latency then invert: achieved throughput = frames/time
                # = 1000/mean(lat), matching the harmonic-mean display metric.
                # EMA-ing per-frame 1000/lat directly is biased high (Jensen).
                L_ema = args.beta * L_ema + (1 - args.beta) * lat
                fps_ema = 1000.0 / L_ema
                fps_tgt = 1000.0 / Ltgt[t]
                # flush integrator on any significant target change (both sawtooth directions)
                if t > 0 and abs(fps_tgt - 1000.0 / Ltgt[t - 1]) > 1.0:
                    integ = 0.0
                # FPS is inverse of latency: higher FPS = more BASE = higher tau
                # so error sign is flipped vs latency mode
                e = fps_ema - fps_tgt   # positive when too fast → need more SUPER → lower tau
            else:
                # latency (ms) or energy (mJ): cost is linear in SUPER rate, same law.
                # Flush threshold scales with the band so it works in both units.
                L_ema = args.beta * L_ema + (1 - args.beta) * lat
                if t > 0 and (Ltgt[t] - Ltgt[t - 1]) < -0.05 * (hi - lo):
                    integ = 0.0
                e = Ltgt[t] - L_ema

            integ += e
            # fps/energy mode: feedforward tau from the known target + PI residual correction
            base_tau = tau_ff[t] if args.mode in ("fps", "energy") else tau0
            tau_un = base_tau - kp * e - ki * integ
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
            if args.mode == "fps":
                # Generate target in FPS space, convert to latency for PI controller
                fps_tgt = target_schedule(bkind, n, fps_lo, fps_hi)
                Ltgt = 1000.0 / fps_tgt
            else:
                Ltgt = target_schedule(bkind, n, lo, hi)
            t_cell = time.perf_counter()
            realized, super_rate = online_loop(meta[fam]["order"], Ltgt, fam)
            Ltgt = Ltgt[:len(realized)]   # match trimmed realized length
            print(f"    [{fam}/{bkind}] {len(realized)} frames in {time.perf_counter()-t_cell:.1f}s",
                  flush=True)
            results[f"{fam}/{bkind}"] = {"super_rate": super_rate,
                                         "realized": realized.tolist(),
                                         "target": Ltgt.tolist()}
            print(f"{fam:<14}{bkind:<10}{super_rate*100:>8.1f}%", flush=True)

    dump = {"l_base": l_base, "l_super": l_super, "win": args.win,
            "kp": args.kp, "ki": args.ki, "mode": args.mode,
            "fam_order": fams, "families": families_meta, "cells": results}
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
