"""Unified, fair on-device (RTX 3090) efficiency benchmark for Table 3 (tab:main),
measured IDENTICALLY for KITTI / BDD100K / Waymo.

For each dataset we measure, on the SAME number of real frames (--n, default 1000)
at the dataset's native resolution, with a common warm-up excluded:
  * BASE  anchor: latency / FPS / energy with all skippable layers skipped,
  * SUPER anchor: latency / FPS / energy with none skipped,
  * router forward overhead (added to every routed frame).
Latency = preprocess+inference+postprocess (Ultralytics result.speed); energy is the
GPU power (pynvml) integrated over that window. Per-frame cost is memoryless, so a
routing policy with realized SUPER-usage p has
  latency(p) = (1-p)*lat_base + p*lat_super + lat_router,   (FPS = 1000/latency)
  energy(p)  = (1-p)*eng_base  + p*eng_super,
exactly how every row of the table is built. AP / SUPER% / GFLOPs come from each
dataset's video Pareto curve (device-independent); only latency/FPS/energy are
measured here. Emits the per-dataset LaTeX rows and a JSON dump of the anchors.

    python -m method02_advantage_regress_tinyConv.bench_device --n 1000
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ultralytics import YOLO

from method02_advantage_regress_tinyConv.build_cache import num_skippable, parse_grid
from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

# name, weight, images_dir, (H,W), router grid, curve json, policy-family regex
DATASETS = [
    ("KITTI", "runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt",
     "/media/data/kitti_yolo/images", (384, 1248), "8",
     OUT / "kitti/eval/video_curve_main_both_g2.json", r"policy_both_s(\d+)"),
    ("BDD100K", "finetuned_bdd100k/anydepth_best.pt",
     "/media/data/bdd100k_yolo/val/images", (720, 1280), "2",
     OUT / "bdd100k/eval/video_curve_full_aligned.json", r"policy_bn(\d+)"),
    ("Waymo", "finetuned_waymo/best.pt",
     "/media/data/waymo_yolo/val/images", (1280, 1920), "2",
     OUT / "waymo/eval_both/video_curve.json", r"policy_seed(\d+)"),
]


class Power:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit(); self.h = pynvml.nvmlDeviceGetHandleByIndex(0); self.p = pynvml
        except Exception as e:
            print(f"[!] pynvml unavailable ({e}); energy=0"); self.p = None

    def mw(self):
        return float(self.p.nvmlDeviceGetPowerUsage(self.h)) if self.p else 0.0


def list_images(root, n):
    root = Path(root)
    files = sorted(root.glob("*.jpg")) or sorted(root.glob("*.png"))
    if not files:                                  # nested per-sequence dirs (Waymo)
        files = sorted(root.rglob("*.jpg")) or sorted(root.rglob("*.png"))
    return [str(f) for f in files[:n]]


def bench_path(yolo, imgs, skip, imgsz, conf, power, warmup):
    lat, eng = [], []
    for i, im in enumerate(imgs):
        p0 = power.mw()
        r = yolo.predict(source=im, imgsz=tuple(imgsz), conf=conf, iou=0.7,
                         skip=skip, verbose=False, device=yolo.device)[0]
        p1 = power.mw()
        ms = r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]
        if i >= warmup:
            lat.append(ms); eng.append(0.5 * (p0 + p1) * (ms / 1000.0))   # mW*s = mJ
    return float(np.mean(lat)), 1000.0 / float(np.mean(lat)), float(np.mean(eng))


def router_overhead(yolo, dev, imgsz, grid):
    from method02_advantage_regress_tinyConv.feature_tap import STATE_LAYERS, INPUT_LEVEL_LAYERS
    from method02_advantage_regress_tinyConv.build_cache import grids, forward_path
    N = yolo.model.num_skippable_layers
    net = PolicyNetwork(group_dim=64, path_dim=8, hidden_dim=128,
                        feat="input", norm="batch", dropout=0.0).to(dev).eval()
    cap = {}
    hs = [yolo.model.model[l].register_forward_hook(
        lambda m, i, o, l=l: cap.__setitem__(l, o)) for l in STATE_LAYERS]
    rh = (imgsz[0] + 31) // 32 * 32; rw = (imgsz[1] + 31) // 32 * 32
    img0 = torch.zeros(1, 3, rh, rw, device=dev)
    with torch.no_grad():
        forward_path(yolo.model, img0, [True] * N, yolo.model.model[-1])
        gin = grids(cap, INPUT_LEVEL_LAYERS, parse_grid(grid))
        z = torch.zeros(1, dtype=torch.long, device=dev)
        net(gin, None, z); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            net.logit(gin, None, z)
        torch.cuda.synchronize()
        rt = (time.perf_counter() - t0) / 200 * 1000.0
    for h in hs:
        h.remove()
    return rt


def curve_points(curve_json, pol_re, metric="map"):
    """Return (endpoints, rows). endpoints: name->(super%,gflops,AP). rows: list of
    (super%, gflops, AP) for the policy budget sweep, averaged over seeds."""
    d = json.loads(Path(curve_json).read_text())
    rows = d["rows"]; pol_re = re.compile(pol_re)
    ends = {}
    pol = defaultdict(list)
    for r in rows:
        if r["name"] in ("always_base", "always_super"):
            ends[r["name"]] = (r["super_rate"] * 100, r["gflops"], r[metric] * 100)
        elif pol_re.fullmatch(r.get("family", "")):
            k = r["budget"] if r.get("budget") is not None else r["thres"]
            pol[k].append((r["super_rate"] * 100, r["gflops"], r[metric] * 100))
    pts = []
    for k in sorted(pol):
        a = np.array(pol[k]); pts.append((a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()))
    pts.sort()
    return ends, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--metric", default="map", choices=["map", "map50"])
    ap.add_argument("--dump", default=str(OUT / "device_table.json"))
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    power = Power()
    result = {}

    for name, weight, images, imgsz, grid, curve, pol_re in DATASETS:
        if not Path(weight).exists():
            print(f"[!] {name}: weight not found ({weight}); skip"); continue
        imgs = list_images(images, args.n)
        if not imgs:
            print(f"[!] {name}: no images under {images}; skip"); continue
        yolo = YOLO(weight, task="detect"); yolo.model.to(dev).eval()
        yolo.model.num_skippable_layers = num_skippable(yolo.model)
        N = yolo.model.num_skippable_layers
        print(f"\n[*] {name}: {len(imgs)} frames @ {imgsz}, warmup {args.warmup}")
        lb, fb, eb = bench_path(yolo, imgs, [True] * N, imgsz, args.conf, power, args.warmup)
        ls, fs, es = bench_path(yolo, imgs, [False] * N, imgsz, args.conf, power, args.warmup)
        rt = router_overhead(yolo, dev, imgsz, grid)
        print(f"    BASE : {lb:6.2f} ms  {fb:5.1f} fps  {eb:8.1f} mJ")
        print(f"    SUPER: {ls:6.2f} ms  {fs:5.1f} fps  {es:8.1f} mJ   router +{rt:.3f} ms")

        ends, pts = curve_points(curve, pol_re, args.metric)
        result[name] = {"lat_base": lb, "lat_super": ls, "eng_base": eb,
                        "eng_super": es, "router_ms": rt, "n": len(imgs),
                        "imgsz": imgsz, "endpoints": ends, "points": pts}

        # ---- LaTeX rows ----
        def blend(p):                       # p in [0,1]
            lat = (1 - p) * lb + p * ls + rt
            return lat, 1000.0 / lat, (1 - p) * eb + p * es

        def row(label, sp, gf, ap):
            lat, fps, en = blend(sp / 100.0)
            return (f"{label} & {sp:.1f} & {gf:.2f} & {ap:.1f} & "
                    f"{lat:.2f} & {fps:.1f} & {en:.0f}\\\\")

        print(f"\n  % ---- {name} table rows ----")
        sb, gb, ab = ends["always_base"]; ss, gs, asu = ends["always_super"]
        print("  " + row(r"\textsc{always-base}", sb, gb, ab))
        for sp, gf, ap in pts:
            print("  " + row(f"{sp:.0f}\\%", sp, gf, ap))
        print("  " + row(r"\textsc{always-super}", ss, gs, asu))

    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.dump, "w"), indent=2)
    print(f"\n[*] anchors -> {args.dump}")


if __name__ == "__main__":
    main()
