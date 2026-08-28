"""On-device (RTX 3090) latency/energy anchors for the BDD100K depth-routing table.

Mirrors the KITTI device measurement: per frame we sum the detector's
preprocess+inference+postprocess time (Ultralytics `result.speed`) and integrate
GPU power (pynvml) over that window. We measure the two path anchors -- BASE
(skip all) and SUPER (skip none) -- plus the router's own forward overhead, on
real BDD100K val frames at 720x1280 with a warm-up phase excluded.

The per-frame cost is memoryless, so any routing threshold with realized
SUPER-usage rate p has  cost(p) = (1-p)*cost_base + p*cost_super + cost_router,
exactly as the KITTI table is built. This script prints the anchors and a small
blended table for the SUPER-rates that appear on the BDD Pareto curve.

    python -m analysis.bench_device_bdd \
        --weight results/step1_finetune/weights/bdd100k/best.pt \
        --images /media/data/bdd100k_yolo/val/images --n 600 --warmup 40
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from ultralytics import YOLO

from step2_train_router.build_cache import num_skippable, parse_grid
from router.router_net import RouterNetwork


class Power:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.p = pynvml
        except Exception as e:
            print(f"[!] pynvml unavailable ({e}); energy=0"); self.p = None

    def mw(self):
        return float(self.p.nvmlDeviceGetPowerUsage(self.h)) if self.p else 0.0


def bench_path(yolo, imgs, skip, imgsz, conf, power, warmup):
    lat, eng = [], []
    for i, im in enumerate(imgs):
        p0 = power.mw()
        r = yolo.predict(source=im, imgsz=tuple(imgsz), conf=conf, iou=0.7,
                         skip=skip, verbose=False, device=yolo.device)[0]
        p1 = power.mw()
        ms = r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]
        if i >= warmup:
            lat.append(ms)
            eng.append(0.5 * (p0 + p1) * (ms / 1000.0))  # mW * s = mJ
    lat, eng = np.array(lat), np.array(eng)
    return lat.mean(), 1000.0 / lat.mean(), eng.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="results/step1_finetune/weights/bdd100k/best.pt")
    ap.add_argument("--images", default="/media/data/bdd100k_yolo/val/images")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--grid", default="2")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.weight, task="detect")
    yolo.model.to(dev).eval()
    yolo.model.num_skippable_layers = num_skippable(yolo.model)
    N = yolo.model.num_skippable_layers
    skip_base, skip_super = [True] * N, [False] * N

    files = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    if not files:
        files = sorted(Path(args.images).glob("*.png"))[: args.n]
    imgs = [str(f) for f in files]
    print(f"[*] {len(imgs)} frames, warmup {args.warmup}, imgsz {args.imgsz}")

    power = Power()
    lb, fb, eb = bench_path(yolo, imgs, skip_base, args.imgsz, args.conf, power, args.warmup)
    ls, fs, es = bench_path(yolo, imgs, skip_super, args.imgsz, args.conf, power, args.warmup)

    # router overhead: time net.logit on a representative input grid (G x G)
    G = parse_grid(args.grid)
    net = RouterNetwork(group_dim=64, path_dim=8, hidden_dim=128,
                        feat="input", norm="batch", dropout=0.0).to(dev).eval()
    Cin = sum(c for c in (256, 512, 512))  # placeholder; materialized below
    # materialize lazily with a dummy of the real channel count by one base forward hook
    from router.feature_tap import STATE_LAYERS, INPUT_LEVEL_LAYERS
    from step2_train_router.build_cache import grids, forward_path
    cap = {}
    hs = [yolo.model.model[l].register_forward_hook(
        lambda m, i, o, l=l: cap.__setitem__(l, o)) for l in STATE_LAYERS]
    rh = (args.imgsz[0] + 31) // 32 * 32
    rw = (args.imgsz[1] + 31) // 32 * 32
    img0 = torch.zeros(1, 3, rh, rw, device=dev)
    with torch.no_grad():
        forward_path(yolo.model, img0, skip_base, yolo.model.model[-1])
        gin = grids(cap, INPUT_LEVEL_LAYERS, G)
        z = torch.zeros(1, dtype=torch.long, device=dev)
        net(gin, None, z)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            net.logit(gin, None, z)
        torch.cuda.synchronize()
        rt = (time.perf_counter() - t0) / 200 * 1000.0
    for h in hs:
        h.remove()

    print("\n==== BDD100K device anchors (RTX 3090) ====")
    print(f" base : lat {lb:6.2f} ms  fps {fb:5.1f}  energy {eb:7.1f} mJ")
    print(f" super: lat {ls:6.2f} ms  fps {fs:5.1f}  energy {es:7.1f} mJ")
    print(f" router fwd overhead: {rt:.3f} ms/frame")

    gb, gs = 33.18, 51.72
    print("\n  super%   GFLOPs   Lat(ms)    FPS   Energy(mJ)")
    for p in (0.0, 0.302, 0.534, 1.0):
        lat = (1 - p) * lb + p * ls + rt
        en = (1 - p) * eb + p * es
        g = (1 - p) * gb + p * gs
        print(f"  {p*100:5.1f}   {g:6.2f}   {lat:7.2f}  {1000/lat:5.1f}   {en:8.1f}")


if __name__ == "__main__":
    main()
