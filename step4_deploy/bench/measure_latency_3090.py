"""Measure base/super latency and energy on RTX 3090 TRT, then print the
Pareto table in the same format as the Jetson table, using AP/GFLOPs/super_rate
from existing eval_video JSON files.

Usage:
    /home/hslee/anaconda3/envs/yolov12/bin/python \
        -m step4_deploy.bench.measure_latency_3090
    /home/hslee/anaconda3/envs/yolov12/bin/python \
        -m step4_deploy.bench.measure_latency_3090 --fp32
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# ── TRT setup ────────────────────────────────────────────────────────────────
_SP = Path(sys.executable).parent.parent / "lib/python3.11/site-packages"
if str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))

import tensorrt as trt
import pynvml

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "kitti": dict(
        eng_dir  = "results/step4_deploy/onnx/kitti_3090",
        json_path= "results/step3_eval/kitti/eval/video_curve_main_both_g2.json",
        fam_re   = r"router_both_s\d+",
        imgsz    = (384, 1248),
    ),
    "bdd100k": dict(
        eng_dir  = "results/step4_deploy/onnx/bdd_pooled_3090",
        json_path= "results/step3_eval/bdd100k/eval/video_curve_archabl.json",
        fam_re   = r"router_bn\d+",
        imgsz    = (736, 1280),
    ),
    "waymo": dict(
        eng_dir  = "results/step4_deploy/onnx/waymo_3090",
        json_path= "results/step3_eval/waymo/eval_both/video_curve.json",
        fam_re   = r"router_seed\d+",
        imgsz    = (1280, 1920),
    ),
}

N_WARMUP = 50
N_MEASURE = 200


def load_engine(path):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(open(path, "rb").read())


def measure_latency_energy(engine, imgsz, nvml_h, n_warmup, n_measure, device="cuda:0"):
    """Returns (mean_latency_ms, mean_energy_mJ)."""
    ctx = engine.create_execution_context()
    dev = torch.device(device)

    # allocate buffers
    io = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = torch.float16
        t = torch.zeros(shape, dtype=dtype, device=dev)
        io[name] = t
        ctx.set_tensor_address(name, t.data_ptr())

    # set input shape for images tensor
    if "images" in io:
        ctx.set_input_shape("images", (1, 3, imgsz[0], imgsz[1]))
        io["images"] = torch.rand(1, 3, imgsz[0], imgsz[1], dtype=torch.float16, device=dev)
        ctx.set_tensor_address("images", io["images"].data_ptr())

    stream = torch.cuda.Stream(device=dev)

    # warmup
    for _ in range(n_warmup):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize(dev)

    # measure
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    lats, energies = [], []
    for _ in range(n_measure):
        pw_before = pynvml.nvmlDeviceGetPowerUsage(nvml_h) / 1000.0  # mW -> W
        start_ev.record(stream)
        ctx.execute_async_v3(stream.cuda_stream)
        end_ev.record(stream)
        torch.cuda.synchronize(dev)
        lat = start_ev.elapsed_time(end_ev)  # ms
        pw_after = pynvml.nvmlDeviceGetPowerUsage(nvml_h) / 1000.0
        pw = 0.5 * (pw_before + pw_after)
        lats.append(lat)
        energies.append(pw * lat)  # W * ms = mJ

    return float(np.mean(lats)), float(np.mean(energies))


def pareto_rows(json_path, fam_re):
    """Return list of (super_rate, map50_95, gflops) sorted by super_rate,
    including always_base and always_super. Averages over seeds/budgets."""
    d = json.loads(Path(json_path).read_text())
    gb, gs = d["gflops_base"], d["gflops_super"]
    pat = re.compile(fam_re)

    anchors = {}
    buckets = defaultdict(list)  # budget -> list of (super_rate, map)

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

    # deduplicate by super_rate (keep unique, sorted)
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
    ap.add_argument("--fp32", action="store_true", help="use FP32 engines instead of FP16")
    ap.add_argument("--fp32-shuffle", action="store_true", help="use fp16_shfl32 mixed engines")
    args = ap.parse_args()
    if args.fp32:
        prec = "fp32"
    elif args.fp32_shuffle:
        prec = "fp16_shfl32"
    else:
        prec = "fp16"

    pynvml.nvmlInit()
    gpu_idx = 0
    nvml_h = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    dev = "cuda:0"

    for ds_name, cfg in DATASETS.items():
        eng_dir = Path(cfg["eng_dir"])
        base_eng  = eng_dir / f"base.{prec}.engine"
        super_eng = eng_dir / f"super.{prec}.engine"

        if not base_eng.exists() or not super_eng.exists():
            print(f"[skip] engines not found for {ds_name}")
            continue

        print(f"\n[{ds_name}] measuring latency & energy on RTX 3090 TRT ...", flush=True)
        eng_b = load_engine(base_eng)
        eng_s = load_engine(super_eng)

        l_base, e_base = measure_latency_energy(eng_b, cfg["imgsz"], nvml_h, N_WARMUP, N_MEASURE, dev)
        l_super, e_super = measure_latency_energy(eng_s, cfg["imgsz"], nvml_h, N_WARMUP, N_MEASURE, dev)
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


if __name__ == "__main__":
    main()
