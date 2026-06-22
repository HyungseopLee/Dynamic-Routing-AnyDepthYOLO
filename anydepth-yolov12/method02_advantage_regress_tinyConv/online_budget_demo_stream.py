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

    python -m method02_advantage_regress_tinyConv.online_budget_demo_stream \
        --weight  fig8_jetson/assets/anydepth_bdd_alpha0.2.pt \
        --policy  fig8_jetson/assets/policy_scenario_s0.pt \
        --scenarios fig8_jetson/assets/scenarios.json \
        --mot_root  fig8_jetson/bdd100k_mot/val \
        --dump fig8_jetson/online_budget_demo_jetson.json --out fig8_jetson/fig8_jetson.pdf
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

from method02_advantage_regress_tinyConv.feature_tap import INPUT_LEVEL_LAYERS, STATE_LAYERS  # noqa
from method02_advantage_regress_tinyConv.eval_video_bdd import (  # noqa
    parse_box_track, labeled_frames, load_policy, grid_vec)
from method02_advantage_regress_tinyConv.online_budget_demo import (  # noqa
    OUT, cond_label, target_schedule, render)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default=None, help="AnyDepth .pt (eager backend)")
    ap.add_argument("--base", default=None, help="BASE TRT engine (.engine)")
    ap.add_argument("--super", default=None, help="SUPER TRT engine (.engine)")
    ap.add_argument("--router_engine", default=None, help="single TRT router engine (iv[,pv]->logit)")
    ap.add_argument("--policy", required=True)
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
    args = ap.parse_args()

    if args.replot:
        render(json.loads(Path(args.dump).read_text()), args.win, args.out)
        return

    dev = args.device if torch.cuda.is_available() else "cpu"

    if args.base and args.super:
        # TensorRT backend: BASE/SUPER (and optionally the router) run as TRT engines.
        # CUDA-event timed; correct for feat=both (computes iv AND pv from pooled taps).
        from trt_bench.jetson_budget_track import TRTBackend
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
            x = grid_vec(captured, INPUT_LEVEL_LAYERS, args.grid).unsqueeze(0)
            x = x.mean(dim=(2, 3)) if is_gap else x
            cuda_sync(); t0 = time.perf_counter()
            with torch.no_grad():
                ahat = float(net.logit(x, None, pid[config]))
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

    # ---- warm-up on the first family's first frames: live BASE/SUPER anchors + tau range
    fam0 = next(iter(scen))
    wb, ws, wa = [], [], []
    for i, (bgr, _first) in enumerate(stream_family(meta[fam0]["order"])):
        if i >= args.warmup:
            break
        lb, ab = run_frame(bgr, "base"); wb.append(lb); wa.append(ab)
        ls, as_ = run_frame(bgr, "super"); ws.append(ls); wa.append(as_)
    l_base = float(np.median(wb[len(wb) // 2:]))
    l_super = float(np.median(ws[len(ws) // 2:]))
    lo = l_base + 0.10 * (l_super - l_base)
    hi = l_super - 0.10 * (l_super - l_base)
    TAU_HI = float(max(wa)) + 0.03
    TAU_LO = float(min(wa)) - 0.03
    tau0 = 0.5 * (TAU_HI + TAU_LO)
    # Auto-scale the PI gains to the actuator: tau lives in a band of width
    # (TAU_HI-TAU_LO) but the error is in ms over a span of ~(l_super-l_base). On BDD
    # the advantage range is tiny (~0.08) while latency errors are ~15 ms, so fixed
    # gains would slam tau between the rails (bang-bang). Scaling by the band ratio
    # makes "a full-scale latency error moves tau ~across its band", dataset-robust.
    gscale = (TAU_HI - TAU_LO) / max(l_super - l_base, 1e-6)
    kp = args.kp * gscale; ki = args.ki * gscale
    print(f"[*] measured anchors: BASE={l_base:.2f}  SUPER={l_super:.2f} ms  "
          f"band [{lo:.2f}, {hi:.2f}];  tau in [{TAU_LO:.3f}, {TAU_HI:.3f}]  "
          f"gscale={gscale:.4f} -> kp={kp:.2e} ki={ki:.2e}", flush=True)

    def online_loop(order, Ltgt):
        n = len(Ltgt)
        config = "base"; tau = TAU_HI; integ = 0.0
        L_ema = l_base
        realized = np.empty(n); n_super = 0
        for t, (bgr, first) in enumerate(stream_family(order)):
            if first:
                config = "base"
            lat, ahat = run_frame(bgr, config)
            realized[t] = lat; n_super += (config == "super")
            L_ema = args.beta * L_ema + (1 - args.beta) * lat
            e = Ltgt[t] - L_ema
            # Positional PI with back-calculation anti-windup (un-saturates the instant
            # the error reverses, so a falling budget is tracked rather than stuck).
            integ += e
            tau_un = tau0 - kp * e - ki * integ
            tau = float(np.clip(tau_un, TAU_LO, TAU_HI))
            if ki:
                integ += (tau_un - tau) / ki
            config = "super" if ahat > tau else "base"
        return realized, n_super / n

    fams = [f for f in scen.keys() if not args.families or f in args.families]
    results = {}; families_meta = {}
    print(f"\n{'scenario':<14}{'budget':<10}{'SUPER%':>9}", flush=True)
    for fam in fams:
        families_meta[fam] = {"bounds": meta[fam]["bounds"], "labels": meta[fam]["labels"]}
        n = meta[fam]["n"]
        for bkind in ("step", "sawtooth"):
            Ltgt = target_schedule(bkind, n, lo, hi)
            t_cell = time.perf_counter()
            realized, super_rate = online_loop(meta[fam]["order"], Ltgt)
            print(f"    [{fam}/{bkind}] {n} frames in {time.perf_counter()-t_cell:.1f}s",
                  flush=True)
            results[f"{fam}/{bkind}"] = {"super_rate": super_rate,
                                         "realized": realized.tolist(),
                                         "target": Ltgt.tolist()}
            print(f"{fam:<14}{bkind:<10}{super_rate*100:>8.1f}%", flush=True)

    dump = {"l_base": l_base, "l_super": l_super, "win": args.win,
            "kp": args.kp, "ki": args.ki, "fps": 0.0,
            "fam_order": fams, "families": families_meta, "cells": results}
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dump, open(args.dump, "w"))
    print(f"[*] -> {args.dump}", flush=True)
    render(dump, args.win, args.out)


if __name__ == "__main__":
    main()
