"""Measure base/super latency and energy on RTX 3090 using PyTorch FP16,
then print the Pareto table in the same format as measure_latency_3090.py.

Usage:
    /home/hslee/anaconda3/envs/yolov12/bin/python \
        -m step4_deploy.bench.measure_latency_pytorch
    /home/hslee/anaconda3/envs/yolov12/bin/python \
        -m step4_deploy.bench.measure_latency_pytorch --compile
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import pynvml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from ultralytics import YOLO
from step2_train_router.build_cache import num_skippable

DATASETS = {
    "kitti": dict(
        weight   = "results/step1_finetune/weights/kitti/best.pt",
        json_path= "results/step3_eval/kitti/eval/video_curve_main_both_g2.json",
        fam_re   = r"router_both_s\d+",
        imgsz    = (384, 1248),
    ),
    "bdd100k": dict(
        weight   = "results/step1_finetune/weights/bdd100k/best.pt",
        json_path= "results/step3_eval/bdd100k/eval/video_curve_archabl.json",
        fam_re   = r"router_bn\d+",
        imgsz    = (736, 1280),
    ),
    "waymo": dict(
        weight   = "results/step1_finetune/weights/waymo/best.pt",
        json_path= "results/step3_eval/waymo/eval_both/video_curve.json",
        fam_re   = r"router_seed\d+",
        imgsz    = (1280, 1920),
    ),
}

N_WARMUP  = 50
N_MEASURE = 200


def measure_latency_energy(model, skip, imgsz, nvml_h, n_warmup, n_measure, device):
    h, w = imgsz
    x = torch.zeros(1, 3, h, w, dtype=torch.float16, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model._predict_once(x, skip=skip, return_features=False)
    torch.cuda.synchronize(device)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    lats, energies = [], []

    with torch.no_grad():
        for _ in range(n_measure):
            pw_before = pynvml.nvmlDeviceGetPowerUsage(nvml_h) / 1000.0  # mW -> W
            start_ev.record()
            model._predict_once(x, skip=skip, return_features=False)
            end_ev.record()
            torch.cuda.synchronize(device)
            lat = start_ev.elapsed_time(end_ev)  # ms
            pw_after = pynvml.nvmlDeviceGetPowerUsage(nvml_h) / 1000.0
            lats.append(lat)
            energies.append(0.5 * (pw_before + pw_after) * lat)  # mJ

    return float(np.mean(lats)), float(np.mean(energies))


def pareto_rows(json_path, fam_re):
    d = json.loads(Path(json_path).read_text())
    gb, gs = d["gflops_base"], d["gflops_super"]
    pat = re.compile(fam_re)

    anchors = {}
    buckets = defaultdict(list)

    for r in d["rows"]:
        nm = r["name"]
        if nm in ("always_base", "always_super"):
            anchors[nm] = (r["super_rate"], r["map"] * 100)
            continue
        if pat.fullmatch(r.get("family", "")):
            key = r.get("budget") if r.get("budget") is not None else r.get("thres")
            buckets[key].append((r["super_rate"], r["map"] * 100))

    rows = []
    if "always_base" in anchors:
        sr, ap = anchors["always_base"]
        rows.append((sr, ap, gb + sr * (gs - gb)))
    for key in sorted(buckets.keys()):
        pts = buckets[key]
        sr = np.mean([p[0] for p in pts])
        ap = np.mean([p[1] for p in pts])
        rows.append((sr, ap, gb + sr * (gs - gb)))
    if "always_super" in anchors:
        sr, ap = anchors["always_super"]
        rows.append((sr, ap, gb + sr * (gs - gb)))

    seen = set()
    deduped = []
    for row in sorted(rows, key=lambda x: x[0]):
        key = round(row[0], 3)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="use torch.compile (CUDA Graphs)")
    args = ap.parse_args()

    pynvml.nvmlInit()
    nvml_h = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0")

    backend_tag = "torch.compile+CUDA Graphs" if args.compile else "PyTorch eager"

    for ds_name, cfg in DATASETS.items():
        print(f"\n[{ds_name}] loading model ...", flush=True)
        yolo = YOLO(cfg["weight"], task="detect")
        m = yolo.model.to(device).half().eval()
        if not hasattr(m, "num_skippable_layers"):
            m.num_skippable_layers = num_skippable(m)
        N = m.num_skippable_layers
        skip_base  = [True]  * N
        skip_super = [False] * N

        if args.compile:
            print(f"  compiling model (this may take a minute) ...", flush=True)
            m = torch.compile(m, mode="reduce-overhead", fullgraph=False)

        print(f"  measuring latency & energy ({backend_tag}, FP16) ...", flush=True)
        l_base,  e_base  = measure_latency_energy(m, skip_base,  cfg["imgsz"], nvml_h, N_WARMUP, N_MEASURE, device)
        l_super, e_super = measure_latency_energy(m, skip_super, cfg["imgsz"], nvml_h, N_WARMUP, N_MEASURE, device)
        print(f"  base:  {l_base:.2f} ms  {1000/l_base:.1f} FPS  {e_base:.0f} mJ")
        print(f"  super: {l_super:.2f} ms  {1000/l_super:.1f} FPS  {e_super:.0f} mJ")

        rows = pareto_rows(cfg["json_path"], cfg["fam_re"])
        print(f"\n  {'tau/label':<14} {'Super%':>7} {'GFLOPs':>8} {'AP':>6} {'Lat(ms)':>9} {'FPS':>6} {'E(mJ)':>8}")
        print("  " + "-" * 68)
        for i, (sr, ap, gf) in enumerate(rows):
            lat = l_base + sr * (l_super - l_base)
            fps = 1000.0 / lat
            eng = e_base + sr * (e_super - e_base)
            if i == 0:
                label = "always-base"
            elif i == len(rows) - 1:
                label = "always-super"
            else:
                label = f"τ pt {i}"
            print(f"  {label:<14} {sr*100:>6.1f}% {gf:>8.2f} {ap:>6.1f} {lat:>9.2f} {fps:>6.1f} {eng:>8.0f}")

        del m, yolo
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
