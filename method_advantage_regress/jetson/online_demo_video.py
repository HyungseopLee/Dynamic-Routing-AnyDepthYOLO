"""Adaptive routing demo video: all three scenario families with time-varying FPS target.

Processes all families from scenarios.json (night<->day, city<->highway, clear<->rainy)
in sequence. Each family has a sawtooth FPS target (continuously varying), tracked by
the feedforward + FPS-space PI controller (same law as online_budget_demo_stream
--mode fps). Renders per frame:
  - Bounding boxes + class labels
  - Config badge: BASE (light) / SUPER (heavy)
  - Realized FPS (EMA) and target FPS in top-right
  - FPS history strip at the bottom with target line
  - Scenario label (top-center)

Backends: TRT engines (--base/--super, detections decoded from the engines' "output"
tensor via NMS) or eager PyTorch (--weight). TRT reaches the paper's 190-310 fps range.

Usage (TRT backend, RTX 3090):
    python -m method_advantage_regress.jetson.online_demo_video \
        --base   method_advantage_regress/jetson/onnx/bdd_pooled/base.fp16.engine \
        --super  method_advantage_regress/jetson/onnx/bdd_pooled/super.fp16.engine \
        --router method_advantage_regress/outputs/bdd100k/router_both_0.pt \
        --router_engine method_advantage_regress/jetson/onnx/bdd_pooled/router.fp16.engine \
        --scenarios method_advantage_regress/outputs/bdd100k/scenarios.json \
        --mot_root  /media/data/bdd100k_mot/val \
        --kp 1.0 --ki 0.10 --beta 0.85
        # output auto-named per family: demo_video_<family>_fps<lo>-<hi>.mp4

Usage (PyTorch eager backend):
    python -m method_advantage_regress.jetson.online_demo_video \
        --weight    finetuning_AnyDepthYOLO/weights/bdd100k/best.pt \
        --router    method_advantage_regress/outputs/bdd100k/router_both_0.pt \
        --scenarios method_advantage_regress/outputs/bdd100k/scenarios.json \
        --mot_root  /media/data/bdd100k_mot/val
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import json
import numpy as np
import torch

from ultralytics import YOLO

from method_advantage_regress.router.feature_tap import (
    INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS)
from method_advantage_regress.eval.eval_video import (
    load_router, grid_vec, parse_box_track, labeled_frames)


def all_frames(video_path):
    """Yield every frame from a video file (no annotation filtering)."""
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()
from method_advantage_regress.jetson.online_budget_demo import OUT, cond_label, target_schedule

# Must match the training class order of finetuning_AnyDepthYOLO/weights/bdd100k/best.pt
# (checked via ckpt model.names): NOT the BDD-MOT order.
BDD_CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train",
]
CLASS_COLORS = [
    (0, 255, 127), (0, 200, 255), (255, 80, 80), (80, 80, 255), (255, 160, 0),
    (160, 0, 255), (0, 255, 255), (255, 0, 200), (255, 220, 0), (0, 180, 255),
]
FAMILY_TITLES = {
    "night_dawn":    "Night <-> Daytime",
    "city_highway":  "City <-> Highway",
    "clear_rainy":   "Clear <-> Rainy",
}


def draw_overlay(frame, boxes, config, fps_realized, fps_target,
                 history, scenario_label, n_history=200):
    H, W = frame.shape[:2]

    # bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = float(box[4]); cls = int(box[5])
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        lbl = f"{BDD_CLASSES[cls] if cls < len(BDD_CLASSES) else cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, lbl, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # config badge (top-left)
    badge_color = (90, 90, 90)
    badge_text  = "SUPER (deep)  " if config == "super" else "BASE  (shallow)"
    cv2.rectangle(frame, (8, 8), (240, 50), badge_color, -1)
    cv2.putText(frame, badge_text, (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # scenario label (top-center)
    (tw, th), _ = cv2.getTextSize(scenario_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cx = (W - tw) // 2
    cv2.rectangle(frame, (cx - 6, 8), (cx + tw + 6, 50), (30, 30, 30), -1)
    cv2.putText(frame, scenario_label, (cx, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    # FPS readout (top-right)
    fps_text = f"measured {fps_realized:5.1f} fps  |  target {fps_target:5.1f} fps"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (W - tw - 16, 8), (W - 4, 44), (30, 30, 30), -1)
    cv2.putText(frame, fps_text, (W - tw - 10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # FPS history strip (bottom)
    bar_h = 70
    strip = np.zeros((bar_h, W, 3), dtype=np.uint8)
    all_vals = history + [fps_target]
    fps_lo = min(all_vals) - 3
    fps_hi = max(all_vals) + 3
    span = max(fps_hi - fps_lo, 1.0)

    def to_y(v):
        return int(bar_h - 4 - (v - fps_lo) / span * (bar_h - 8))

    # target line
    ty = to_y(fps_target)
    cv2.line(strip, (0, ty), (W, ty), (60, 60, 60), 2)
    cv2.putText(strip, f"target {fps_target:.0f} fps", (6, max(ty - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # realized history
    n = len(history)
    if n > 1:
        show = history[-min(n, n_history):]
        xs = np.linspace(0, W - 1, len(show)).astype(int)
        for x, v in zip(xs, show):
            vy = to_y(v)
            color = (60, 200, 60) if v >= fps_target - 2 else (60, 60, 220)
            cv2.line(strip, (x, bar_h - 2), (x, vy), color, 1)

    frame[-bar_h:] = strip
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight",     default=None, help="AnyDepth .pt (eager PyTorch backend)")
    ap.add_argument("--base",       default=None, help="BASE TRT engine (.engine)")
    ap.add_argument("--super",      default=None, help="SUPER TRT engine (.engine)")
    ap.add_argument("--router_engine", default=None, help="TRT router engine (iv[,pv]->logit)")
    ap.add_argument("--router",     required=True)
    ap.add_argument("--scenarios",  default=str(OUT / "bdd100k/scenarios.json"))
    ap.add_argument("--mot_root",   default="/media/data/bdd100k_mot/val")
    ap.add_argument("--imgsz",      type=int, nargs=2, default=[736, 1280])
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--grid",       type=int, default=2)
    ap.add_argument("--kp",         type=float, default=1.0)
    ap.add_argument("--ki",         type=float, default=0.10)
    ap.add_argument("--beta",       type=float, default=0.85)
    ap.add_argument("--warmup",     type=int, default=30)
    ap.add_argument("--out_fps",    type=float, default=30.0,
                    help="output video FPS")
    ap.add_argument("--out_dir",    default=str(OUT / "figures"),
                    help="directory for output mp4 files")
    ap.add_argument("--families",   nargs="*", default=None,
                    help="subset of scenario families to render")
    ap.add_argument("--device",     default="cuda:0")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    label_dir = Path(args.mot_root) / "labels"
    video_dir = Path(args.mot_root) / "videos"
    scen = json.loads(Path(args.scenarios).read_text())

    # ---- backend: unified run_frame(bgr, config) -> (lat_ms, ahat, boxes, disp_frame)
    # boxes = [[x1,y1,x2,y2,conf,cls], ...] in disp_frame coordinates.
    if args.base and args.super:
        # TensorRT: engines emit the detection tensor ("output") alongside the router
        # taps; TRTBackend times detector+router, we NMS the detections afterwards
        # (NMS is outside the timed scope, matching online_budget_demo_stream).
        from method_advantage_regress.jetson.jetson_budget_track import TRTBackend
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import ops
        backend = TRTBackend(args.base, args.super, args.router, args.grid, dev,
                             router_engine=args.router_engine)
        _, _, He, We = backend.eng["base"]["inp"].shape   # engine input size is fixed
        letterbox = LetterBox((He, We), auto=False, stride=32)
        disp_size = (We, He)   # (width, height) for VideoWriter
        print(f"[*] TRT backend: {args.base} / {args.super}"
              + (f" + router {args.router_engine}" if args.router_engine else " (eager router)")
              + f"  input {He}x{We}", flush=True)

        def run_frame(bgr, config):
            img = letterbox(image=bgr)
            t = torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()).to(dev)
            x = (t.float() / 255.0).unsqueeze(0)
            ahat, lat = backend.run_frame(x, config)
            pred = backend.eng[config]["out"]["output"].float()
            det = ops.non_max_suppression(pred, conf_thres=args.conf, iou_thres=0.7)[0]
            boxes = det.cpu().numpy().tolist()   # already in letterboxed coords
            return lat, ahat, boxes, img
    else:
        if not args.weight:
            raise SystemExit("either --weight (PyTorch) or --base/--super (TRT) is required")
        yolo = YOLO(args.weight, task="detect")
        yolo.model.to(dev).eval()
        from method_advantage_regress.train.build_cache import num_skippable
        yolo.model.num_skippable_layers = num_skippable(yolo.model)
        N = yolo.model.num_skippable_layers
        skip = {"super": [False] * N, "base": [True] * N}
        net, _ckpt, feat, is_gap = load_router(args.router, dev)
        pid = {"super": torch.ones(1, dtype=torch.long, device=dev),
               "base":  torch.zeros(1, dtype=torch.long, device=dev)}
        captured = {}
        for idx in STATE_LAYERS:
            yolo.model.model[idx].register_forward_hook(
                lambda _m, _i, o, k=idx: captured.__setitem__(k, o))
        disp_size = (args.imgsz[1], args.imgsz[0])

        def cuda_sync():
            if dev.startswith("cuda"):
                torch.cuda.synchronize()

        def run_frame(bgr, config):
            captured.clear()
            r = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                             iou=0.7, skip=skip[config], verbose=False, device=dev)[0]
            cuda_sync()
            lat = r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]
            xi = grid_vec(captured, INPUT_LEVEL_LAYERS, args.grid).unsqueeze(0)
            xp = grid_vec(captured, PRED_LEVEL_LAYERS, args.grid).unsqueeze(0) if feat != "input" else None
            if is_gap:
                xi = xi.mean(dim=(2, 3))
                xp = xp.mean(dim=(2, 3)) if xp is not None else None
            with torch.no_grad():
                ahat = float(net.logit(xi, xp, pid[config]))
            boxes = []
            if r.boxes is not None and len(r.boxes):
                xyxy  = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss  = r.boxes.cls.cpu().numpy().astype(int)
                for j in range(len(xyxy)):
                    boxes.append([*xyxy[j], confs[j], clss[j]])
            return lat, ahat, boxes, cv2.resize(bgr, disp_size)

    def stream_family(order):
        for seg in order:
            first = True
            for bgr in all_frames(video_dir / f"{seg['seq']}.mov"):
                yield bgr, first, cond_label(seg.get("cond", {}))
                first = False

    # warmup on first family (backend-agnostic: alternate BASE/SUPER via run_frame)
    fam0 = next(iter(scen))
    print(f"[*] warming up {args.warmup} frames on {fam0} ...", flush=True)
    wb, ws, wa = [], [], []
    for i, (bgr, _, _) in enumerate(stream_family(scen[fam0])):
        if i >= args.warmup * 2:
            break
        cfg_w = "base" if i % 2 == 0 else "super"
        ms, ahat, _, _ = run_frame(bgr, cfg_w)
        (wb if cfg_w == "base" else ws).append(ms)
        wa.append(ahat)
    l_base  = float(np.median(wb[len(wb) // 2:]))
    l_super = float(np.median(ws[len(ws) // 2:]))

    TAU_HI = float(max(wa)) + 0.03
    TAU_LO = float(min(wa)) - 0.03
    tau0   = 0.5 * (TAU_HI + TAU_LO)
    lo_lat = l_base + 0.10 * (l_super - l_base)
    hi_lat = l_super - 0.10 * (l_super - l_base)
    fps_hi = 1000.0 / lo_lat
    fps_lo = 1000.0 / hi_lat
    # FPS-space PI (matches online_budget_demo_stream --mode fps): error in FPS units,
    # gains auto-scaled by the achievable FPS span
    fps_base = 1000.0 / l_base
    fps_super = 1000.0 / l_super
    gscale = (TAU_HI - TAU_LO) / max(fps_base - fps_super, 1e-6)
    kp = args.kp * gscale
    ki = args.ki * gscale
    print(f"[*] BASE={l_base:.2f}ms SUPER={l_super:.2f}ms  "
          f"FPS range [{fps_lo:.1f}, {fps_hi:.1f}]  "
          f"kp={kp:.2e} ki={ki:.2e}", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fams = [f for f in scen if not args.families or f in args.families]

    for fam in fams:
        order = scen[fam]
        # count total frames (all frames, not just annotated)
        counts = []
        for seg in order:
            cap = cv2.VideoCapture(str(video_dir / f"{seg['seq']}.mov"))
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        n_total = sum(counts)

        # sawtooth FPS target for this family (continuously varying)
        fps_tgt_arr = target_schedule("sawtooth", n_total, fps_lo, fps_hi)
        # convert to latency for PI controller
        L_tgt_arr = 1000.0 / fps_tgt_arr
        # feedforward tau from the known target (same as online_budget_demo_stream):
        # mean latency is linear in SUPER rate p; p maps to the (1-p)-quantile of Ahat
        pct = np.clip((L_tgt_arr - l_base) / max(l_super - l_base, 1e-6), 0.0, 1.0)
        tau_ff = np.clip(np.quantile(wa, 1.0 - pct), TAU_LO, TAU_HI)

        out_path = str(Path(args.out_dir) /
                       f"demo_video_{fam}_fps{fps_lo:.0f}-{fps_hi:.0f}.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.out_fps,
            disp_size,
        )

        # PI state (reset per family)
        config = "base"; integ = 0.0
        L_ema = L_tgt_arr[0]
        fps_history = []

        frame_idx = 0
        print(f"\n[*] rendering {fam}: {n_total} frames -> {out_path}", flush=True)
        for bgr, first, seg_label in stream_family(order):
            if first:
                config = "base"; integ = 0.0

            fps_tgt_now = float(fps_tgt_arr[min(frame_idx, n_total - 1)])

            lat, ahat, boxes, disp = run_frame(bgr, config)

            # PI update in FPS space (throughput = 1000/mean(lat), so EMA the latency
            # and invert; error sign flipped vs latency: too fast -> need more SUPER)
            L_ema = args.beta * L_ema + (1 - args.beta) * lat
            fps_ema = 1000.0 / L_ema
            prev_fps_tgt = float(fps_tgt_arr[max(frame_idx - 1, 0)])
            if frame_idx > 0 and abs(fps_tgt_now - prev_fps_tgt) > 1.0:
                integ = 0.0   # flush on significant target change
            e = fps_ema - fps_tgt_now
            integ += e
            tau_un = float(tau_ff[min(frame_idx, n_total - 1)]) - kp * e - ki * integ
            tau = float(np.clip(tau_un, TAU_LO, TAU_HI))
            if ki:
                integ += (tau_un - tau) / ki
            next_config = "super" if ahat > tau else "base"

            fps_realized = 1000.0 / L_ema
            fps_history.append(fps_realized)

            fam_title = FAMILY_TITLES.get(fam, fam)
            scene_str = f"{fam_title}  [{seg_label}]"
            out_frame = draw_overlay(disp, boxes, config, fps_realized, fps_tgt_now,
                                     fps_history, scene_str)
            writer.write(out_frame)

            config = next_config
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  {fam} frame {frame_idx}/{n_total}  "
                      f"fps_realized={fps_realized:.1f}  fps_target={fps_tgt_now:.1f}  "
                      f"config={config}", flush=True)

        writer.release()
        print(f"[*] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
