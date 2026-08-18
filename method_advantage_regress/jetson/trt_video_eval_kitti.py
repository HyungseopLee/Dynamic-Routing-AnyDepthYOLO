"""Real end-to-end TensorRT video evaluation of depth routing on KITTI-Tracking.

Mirrors trt_video_eval.py (BDD) but for KITTI image sequences and a feat=both
router (which consumes BOTH input-level taps 4,6,8 and pred-level taps 14,17,20).

Per routing threshold tau we stream each KITTI sequence causally ON THE ENGINES:
  frame t: choice = SUPER if (t>0 and Ahat_{t-1} > tau) else BASE
           run ONLY the chosen path's engine -> detections + raw tap maps
           grid-pool the taps (exact training op) -> router (eager) -> Ahat_t
We measure realized SUPER-usage and per-frame latency / FPS / energy for the
detector engine inference + router ONLY (preprocess and NMS excluded, matching
the BDD scope). Each sequence's frames are decoded ONCE and reused across all
taus (decode-once), so the disk/decode cost is not paid 11x.

    python method_advantage_regress/jetson/trt_video_eval_kitti.py \
        --base method_advantage_regress/jetson/onnx/kitti/base.fp16.engine \
        --super method_advantage_regress/jetson/onnx/kitti/super.fp16.engine \
        --router method_advantage_regress/outputs/kitti/router.pt \
        --kitti_root /media/data/kitti-tracking --imgsz 384 1248
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox

from method_advantage_regress.router.feature_tap import (
    INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS)
from method_advantage_regress.eval.eval_video import load_router

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class Engine:
    """Single TRT engine with resident I/O buffers (loaded once)."""

    def __init__(self, path, device):
        with open(path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.out = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dt = trt.nptype(self.engine.get_tensor_dtype(name))
            t = torch.zeros(shape, dtype=getattr(torch, np.dtype(dt).name), device=device)
            self.ctx.set_tensor_address(name, t.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inp = t
            else:
                self.out[name] = t

    def run(self, x, stream):
        self.inp.copy_(x)
        self.ctx.execute_async_v3(stream.cuda_stream)


class Power:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit(); self.h = pynvml.nvmlDeviceGetHandleByIndex(0); self.p = pynvml
        except Exception:
            self.p = None

    def mw(self):
        return float(self.p.nvmlDeviceGetPowerUsage(self.h)) if self.p else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--super", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--kitti_root", default="/media/data/kitti-tracking")
    ap.add_argument("--sequences", type=str, nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1248])
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--taus", type=float, nargs="*",
                    default=[1.00, 0.70, 0.50, 0.40, 0.30, 0.20, 0.10,
                             0.00, -0.10, -0.20, -0.40])
    ap.add_argument("--limit", type=int, default=0, help="number of sequences (0=all)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="method_advantage_regress/jetson/outputs/trt_video_eval_kitti.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    H = (args.imgsz[0] + 31) // 32 * 32
    W = (args.imgsz[1] + 31) // 32 * 32
    lb = LetterBox((H, W), auto=False, stride=32)

    eng = {"base": Engine(args.base, dev), "super": Engine(args.super, dev)}
    net, ckpt, feat, is_gap = load_router(args.router, dev)
    pid = {"base": torch.zeros(1, dtype=torch.long, device=dev),
           "super": torch.ones(1, dtype=torch.long, device=dev)}
    use_pred = feat in ("both", "pred")
    power = Power()

    img_root = Path(args.kitti_root) / "training" / "image_02"
    seqs = args.sequences or sorted(p.name for p in img_root.iterdir() if p.is_dir())
    if args.limit:
        seqs = seqs[:args.limit]

    def frames_of(seq):
        d = img_root / seq
        return sorted(d.glob("*.png")) or sorted(d.glob("*.jpg"))

    def preprocess(bgr):
        img = lb(image=bgr)
        t = torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()).to(dev)
        return (t.float() / 255.0).unsqueeze(0)

    # build lazy router layers once with the real feature dims, then load weights
    in_c = sum(eng["base"].out[f"feat{l}"].shape[1] for l in INPUT_LEVEL_LAYERS)
    pr_c = sum(eng["base"].out[f"feat{l}"].shape[1] for l in PRED_LEVEL_LAYERS)
    with torch.no_grad():
        net(torch.zeros(2, in_c, args.grid, args.grid, device=dev),
            torch.zeros(2, pr_c, args.grid, args.grid, device=dev) if use_pred else None,
            torch.zeros(2, dtype=torch.long, device=dev))
    net.load_state_dict(ckpt["state_dict"]); net.eval()

    @torch.no_grad()
    def run_frame(x, choice, stream):
        # Timed scope: detector engine inference + router only.
        e = eng[choice]
        e.run(x, stream)
        stream.synchronize()
        iv = torch.cat([F.adaptive_avg_pool2d(e.out[f"feat{l}"].float(), args.grid).squeeze(0)
                        for l in INPUT_LEVEL_LAYERS], dim=0).unsqueeze(0)
        pv = (torch.cat([F.adaptive_avg_pool2d(e.out[f"feat{l}"].float(), args.grid).squeeze(0)
                         for l in PRED_LEVEL_LAYERS], dim=0).unsqueeze(0) if use_pred else None)
        return float(net.logit(iv, pv, pid[choice]).view(-1))

    stream = torch.cuda.Stream()
    dummy = torch.zeros(1, 3, H, W, device=dev)
    with torch.cuda.stream(stream):
        for _ in range(30):
            run_frame(dummy, "base", stream); run_frame(dummy, "super", stream)
    torch.cuda.synchronize()

    # accumulate per-tau across all sequences; DECODE-ONCE: read each sequence's
    # frames a single time, then replay all taus over the in-memory frames.
    acc = {tau: {"lat": [], "eng": [], "ns": 0, "n": 0} for tau in args.taus}
    for si, seq in enumerate(seqs):
        fps = frames_of(seq)
        bgrs = [cv2.imread(str(p)) for p in fps]          # decode once
        xs = [preprocess(b) for b in bgrs]                 # preprocess once (excluded from timing)
        del bgrs
        for tau in args.taus:
            a = acc[tau]
            prev_ah, prev_choice = None, None
            with torch.cuda.stream(stream):
                for x in xs:
                    choice = "super" if (prev_choice is not None and prev_ah > tau) else "base"
                    p0 = power.mw(); t0 = time.perf_counter()
                    ah = run_frame(x, choice, stream)
                    dt = (time.perf_counter() - t0) * 1000.0
                    p1 = power.mw()
                    a["lat"].append(dt); a["eng"].append(0.5 * (p0 + p1) * dt / 1000.0)
                    a["ns"] += (choice == "super"); a["n"] += 1
                    prev_ah, prev_choice = ah, choice
        del xs
        print(f"  [{si+1}/{len(seqs)}] seq {seq}: {len(fps)} frames done", flush=True)

    print(f"\n{'tau':>7}{'SUPER%':>9}{'lat(ms)':>9}{'FPS':>7}{'energy(mJ)':>12}")
    rows = []
    for tau in args.taus:
        a = acc[tau]
        lat, en = a["lat"], a["eng"]
        L = float(np.mean(lat[5:])) if len(lat) > 10 else float(np.mean(lat))
        sp = 100.0 * a["ns"] / a["n"]
        E = float(np.mean(en[5:])) if len(en) > 10 else float(np.mean(en))
        print(f"{tau:>7.3f}{sp:>9.1f}{L:>9.2f}{1000.0/L:>7.1f}{E:>12.1f}")
        rows.append({"tau": tau, "super_pct": sp, "lat_ms": L, "fps": 1000.0 / L, "energy_mj": E})

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\n[*] -> {args.out}  ({acc[args.taus[0]]['n']} frames/threshold over {len(seqs)} sequences)")


if __name__ == "__main__":
    main()
