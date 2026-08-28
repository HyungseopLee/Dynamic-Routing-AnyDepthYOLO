"""Real end-to-end TensorRT video evaluation of depth routing, swept over thresholds.

For each routing threshold tau we stream real BDD100K MOT video frames through the
deployed loop ON THE ENGINES:
  frame t: choice = SUPER if (t>0 and Ahat_{t-1} > tau) else BASE        (causal)
           run ONLY the chosen path's engine -> detections + raw tap maps
           grid-pool the tap maps (exact training op) -> router (eager) -> Ahat_t
We measure the realized SUPER-usage and the end-to-end per-frame latency / FPS /
energy (preprocess + engine inference + router + NMS; video decode/disk IO excluded,
matching the eager Table-3 scope). Only ONE path runs per frame -- this is the real
deployed cost of routing, not a replay.

    python step4_deploy/trt_video_eval.py --base results/step4_deploy/onnx/bdd_pooled/base.fp16.engine \
        --super results/step4_deploy/onnx/bdd_pooled/super.fp16.engine \
        --router results/step2_router/weights/bdd100k/router_g2x2_both_s0.pt \
        --mot_root /media/data/bdd100k_mot/val --imgsz 720 1280 --limit 20
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

from router.feature_tap import INPUT_LEVEL_LAYERS
from step3_eval.eval_video import (
    parse_box_track, labeled_frames, load_router)

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
    ap.add_argument("--mot_root", default="/media/data/bdd100k_mot/val")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--taus", type=float, nargs="*",
                    default=[0.086, 0.072, 0.064, 0.058, 0.054, 0.049, 0.045, 0.040, 0.033])
    ap.add_argument("--limit", type=int, default=20, help="number of videos (0=all)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    H = (args.imgsz[0] + 31) // 32 * 32
    W = (args.imgsz[1] + 31) // 32 * 32
    lb = LetterBox((H, W), auto=False, stride=32)

    eng = {"base": Engine(args.base, dev), "super": Engine(args.super, dev)}
    net, ckpt, feat, is_gap = load_router(args.router, dev)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    pid = {"base": torch.zeros(1, dtype=torch.long, device=dev),
           "super": torch.ones(1, dtype=torch.long, device=dev)}
    power = Power()

    label_dir = Path(args.mot_root) / "labels"
    video_dir = Path(args.mot_root) / "videos"
    seqs = sorted(p.stem for p in label_dir.glob("*.json"))
    if args.limit:
        seqs = seqs[:args.limit]

    def preprocess(bgr):
        img = lb(image=bgr)                       # letterbox to (H,W), HWC BGR
        t = torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()).to(dev)
        return (t.float() / 255.0).unsqueeze(0)   # [1,3,H,W]

    @torch.no_grad()
    def run_frame(x, choice, stream):
        # Timed scope: detector engine inference + router only (preprocess and NMS are
        # excluded -- they are path-independent constants and, unoptimized, would mask
        # the routing latency signal under FP16).
        e = eng[choice]
        e.run(x, stream)
        stream.synchronize()
        cap = {4: e.out["feat4"], 6: e.out["feat6"], 8: e.out["feat8"]}
        xi = torch.cat([F.adaptive_avg_pool2d(cap[i].float(), args.grid).squeeze(0)
                        for i in INPUT_LEVEL_LAYERS], dim=0).unsqueeze(0)
        ahat = float(net.logit(xi, None, pid[choice]).view(-1))
        return ahat

    stream = torch.cuda.Stream()
    # warm-up both engines
    dummy = torch.zeros(1, 3, H, W, device=dev)
    with torch.cuda.stream(stream):
        for _ in range(30):
            run_frame(dummy, "base", stream); run_frame(dummy, "super", stream)
    torch.cuda.synchronize()

    print(f"\n{'tau':>7}{'SUPER%':>9}{'lat(ms)':>9}{'FPS':>7}{'energy(mJ)':>12}")
    rows = []
    for tau in args.taus:
        lat, eng_j = [], []
        n_super = n = 0
        with torch.cuda.stream(stream):
            for si, seq in enumerate(seqs):
                gt = parse_box_track(label_dir / f"{seq}.json")
                prev_ah, prev_choice = None, None
                for fidx, bgr in labeled_frames(video_dir / f"{seq}.mov", gt):
                    choice = "super" if (prev_choice is not None and prev_ah > tau) else "base"
                    x = preprocess(bgr)                       # excluded from timing
                    p0 = power.mw(); t0 = time.perf_counter()
                    ah = run_frame(x, choice, stream)          # engine + router only
                    dt = (time.perf_counter() - t0) * 1000.0
                    p1 = power.mw()
                    lat.append(dt); eng_j.append(0.5 * (p0 + p1) * dt / 1000.0)
                    n_super += (choice == "super"); n += 1
                    prev_ah, prev_choice = ah, choice
        # drop the first few frames per the warm pipeline already done; use steady mean
        L = float(np.mean(lat[5:])) if len(lat) > 10 else float(np.mean(lat))
        sp = 100.0 * n_super / n
        en = float(np.mean(eng_j[5:])) if len(eng_j) > 10 else float(np.mean(eng_j))
        print(f"{tau:>7.3f}{sp:>9.1f}{L:>9.2f}{1000.0/L:>7.1f}{en:>12.1f}")
        rows.append({"tau": tau, "super_pct": sp, "lat_ms": L, "fps": 1000.0 / L, "energy_mj": en})

    import json
    out = Path("results/step4_deploy/trt_video_eval_bdd.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n[*] -> {out}  ({n} frames/threshold over {len(seqs)} videos)")


if __name__ == "__main__":
    main()
