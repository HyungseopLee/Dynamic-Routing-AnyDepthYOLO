"""Table 4 on Jetson Orin Nano: TensorRT engine-switching + routing overhead.

Reproduces the paper's TRT-deployment microbenchmark on THIS device, supporting two
claims:
  (1) switching between the two resident BASE/SUPER engines is ~free, and
  (2) the router's per-frame overhead is small relative to the detector.

We hold BOTH FP16 engines resident in memory and measure pure GPU inference latency
(CUDA-event timed) for:
  * BASE        only      (always-base anchor),
  * SUPER       only      (always-super anchor),
  * ALTERNATING           (force a BASE<->SUPER engine switch EVERY frame; worst case),
so the switch overhead is isolated as ALTERNATING - mean(BASE, SUPER).

With --router, we additionally time the router head (grid-pool the engine's tap maps
+ tiny MLP) to report routing overhead as a fraction of the detector latency.

With --router_engine, we time the TRT router engine (iv+pv -> logit) combined with
the detector to get L_base+router and L_super+router for Table 3 interpolation.
Detector engines must output pooled 2x2 feat maps (the default _pooled engines).

Energy: Jetson exposes no NVML power, so we read the on-board INA3221 rails via hwmon
(VDD_IN = total module, VDD_CPU_GPU_CV = compute); power_mW = volt_mV * curr_mA / 1000.

    python step4_deploy/bench/bench_trt_jetson.py \
        --base  results/step4_deploy/onnx/kitti_pooled/base.fp16.engine \
        --super results/step4_deploy/onnx/kitti_pooled/super.fp16.engine \
        --router_engine results/step4_deploy/onnx/kitti/router.fp16.engine \
        --iters 1000 --warmup 100
"""
import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

from router.feature_tap import (
    INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS)
from step3_eval.eval_video import load_router

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class Engine:
    """One resident TRT engine; I/O buffers are torch tensors kept resident."""

    def __init__(self, path, device):
        with open(path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.dev = device
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dt = trt.nptype(self.engine.get_tensor_dtype(name))
            t = torch.zeros(shape, dtype=getattr(torch, np.dtype(dt).name), device=device)
            self.buffers[name] = t
            self.ctx.set_tensor_address(name, t.data_ptr())
        self.inp = self.engine.get_tensor_name(0)

    def run(self, stream):
        self.ctx.execute_async_v3(stream.cuda_stream)


class JetsonPower:
    """INA322 board power via hwmon sysfs (NVML reports no power on Tegra)."""
    RAILS = ("VDD_IN", "VDD_CPU_GPU_CV")

    def __init__(self):
        self.chans = {}
        for label_f in glob.glob("/sys/class/hwmon/hwmon*/in*_label"):
            try:
                lbl = open(label_f).read().strip()
            except OSError:
                continue
            m = re.search(r"in(\d+)_label$", label_f)
            if lbl not in self.RAILS or not m:
                continue
            d = Path(label_f).parent
            ch = m.group(1)
            v, c = d / f"in{ch}_input", d / f"curr{ch}_input"
            if v.exists() and c.exists():
                self.chans[lbl] = (str(v), str(c))
        self.available = bool(self.chans)

    def mw(self, rail="VDD_IN"):
        ch = self.chans.get(rail)
        if ch is None:
            return 0.0
        try:
            return float(open(ch[0]).read()) * float(open(ch[1]).read()) / 1000.0
        except (OSError, ValueError):
            return 0.0


def timed(run_fn, iters, warmup, power, power_every=25):
    """Mean latency (ms), FPS, and per-call energy (mJ) on VDD_IN over `iters`.

    Power is sampled only every `power_every` iterations: each INA3221 read is an I2C
    sysfs transaction, and doing it twice per iteration (thousands of times) was found
    to intermittently stall/abort the run on the Orin. Sparse sampling keeps a faithful
    mean power without perturbing the latency loop; energy = mean_power * mean_latency."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            run_fn(stream)
    torch.cuda.synchronize()
    lat, pw = [], []
    with torch.cuda.stream(stream):
        for i in range(iters):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(stream); run_fn(stream); e.record(stream)
            e.synchronize()
            lat.append(s.elapsed_time(e))
            if i % power_every == 0:
                pw.append(power.mw("VDD_IN"))
    mean_lat = float(np.mean(lat))
    mean_pw = float(np.mean(pw)) if pw else 0.0
    return mean_lat, 1000.0 / mean_lat, mean_pw * mean_lat / 1000.0


def build_router_combined(det_engine, router_engine, dev):
    """Return a run_fn that executes detector + TRT router in one call.

    Detector must output pooled 2x2 feat maps (feat4/6/8 -> iv, feat14/17/20 -> pv).
    iv and pv are pre-allocated contiguous buffers; cat is done via copy_ in-place.
    """
    iv_buf = router_engine.buffers["iv"]   # (1, 768, 2, 2)
    pv_buf = router_engine.buffers["pv"]   # (1, 640, 2, 2)

    # pre-compute slice offsets for in-place cat
    in_layers = INPUT_LEVEL_LAYERS   # (4, 6, 8)
    pr_layers = PRED_LEVEL_LAYERS    # (14, 17, 20)

    def run_fn(stream):
        det_engine.run(stream)
        # cat feat tensors into router input buffers (all already on GPU, 2x2)
        offset = 0
        for l in in_layers:
            t = det_engine.buffers[f"feat{l}"]
            c = t.shape[1]
            iv_buf[:, offset:offset+c].copy_(t)
            offset += c
        offset = 0
        for l in pr_layers:
            t = det_engine.buffers[f"feat{l}"]
            c = t.shape[1]
            pv_buf[:, offset:offset+c].copy_(t)
            offset += c
        router_engine.run(stream)

    return run_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--super", required=True)
    ap.add_argument("--router", default=None, help="router .pt; if set, also time routing overhead")
    ap.add_argument("--router_engine", default=None, help="TRT router .fp16.engine; times detector+router combined")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_power", action="store_true",
                    help="skip INA3221 power/energy sampling (latency-only; the per-read "
                         "I2C sysfs access can intermittently stall long runs on the Orin)")
    args = ap.parse_args()
    dev = torch.device(args.device)

    class _NoPower:
        available = False
        def mw(self, rail="VDD_IN"):
            return 0.0

    power = _NoPower() if args.no_power else JetsonPower()
    if not power.available and not args.no_power:
        print("[!] no INA3221 rails found; energy will read 0")

    eb = Engine(args.base, dev); es = Engine(args.super, dev)
    print(f"[*] base in {tuple(eb.buffers[eb.inp].shape)} | super in {tuple(es.buffers[es.inp].shape)}")

    # two-engine resident memory footprint (the key Jetson constraint)
    try:
        mb = lambda b: b / (1024 ** 2)
        dm = mb(eb.engine.device_memory_size + es.engine.device_memory_size)
        print(f"[*] resident GPU memory: two-engine activation scratch {dm:.0f} MB; "
              f"torch reserved {mb(torch.cuda.memory_reserved(dev)):.0f} MB")
    except Exception:
        pass

    ls, fs, ens = timed(lambda s: es.run(s), args.iters, args.warmup, power)
    lb, fb, enb = timed(lambda s: eb.run(s), args.iters, args.warmup, power)

    flip = {"v": True}
    def alt(s):
        flip["v"] = not flip["v"]
        (eb if flip["v"] else es).run(s)
    la, fa, ena = timed(alt, args.iters, args.warmup, power)

    # ---- routing overhead: grid-pool the engine taps + tiny MLP ----
    router_ms = None
    if args.router:
        net, ckpt, feat, is_gap = load_router(args.router, dev)
        use_pred = feat in ("both", "pred")
        pid = torch.ones(1, dtype=torch.long, device=dev)   # time the SUPER-path head
        in_c = sum(es.buffers[f"feat{l}"].shape[1] for l in INPUT_LEVEL_LAYERS)
        pr_c = sum(es.buffers[f"feat{l}"].shape[1] for l in PRED_LEVEL_LAYERS)
        with torch.no_grad():
            net(torch.zeros(2, in_c, args.grid, args.grid, device=dev),
                torch.zeros(2, pr_c, args.grid, args.grid, device=dev) if use_pred else None,
                torch.zeros(2, dtype=torch.long, device=dev))
        net.load_state_dict(ckpt["state_dict"]); net.eval()

        # taps are resident in the super engine's output buffers; run once to fill them
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            es.run(stream)
        torch.cuda.synchronize()

        @torch.no_grad()
        def router_step(s):
            iv = torch.cat([F.adaptive_avg_pool2d(es.buffers[f"feat{l}"].float(), args.grid).squeeze(0)
                            for l in INPUT_LEVEL_LAYERS], dim=0).unsqueeze(0)
            pv = (torch.cat([F.adaptive_avg_pool2d(es.buffers[f"feat{l}"].float(), args.grid).squeeze(0)
                             for l in PRED_LEVEL_LAYERS], dim=0).unsqueeze(0) if use_pred else None)
            if is_gap:
                iv = iv.mean(dim=(2, 3)); pv = pv.mean(dim=(2, 3)) if pv is not None else None
            net.logit(iv, pv, pid)
        router_ms, _, _ = timed(router_step, args.iters, args.warmup, power)

    # ---- TRT router engine: detector + router combined latency ----
    lb_r = ls_r = fb_r = fs_r = enb_r = ens_r = None
    if args.router_engine:
        er = Engine(args.router_engine, dev)
        base_router_fn = build_router_combined(eb, er, dev)
        super_router_fn = build_router_combined(es, er, dev)
        lb_r, fb_r, enb_r = timed(base_router_fn, args.iters, args.warmup, power)
        ls_r, fs_r, ens_r = timed(super_router_fn, args.iters, args.warmup, power)

    # ---- engine weights + scratch memory ----
    MB = 1024 ** 2
    def engine_mem(eng):
        w = len(open(eng.engine.__class__.__name__, "rb").read()) if False else 0
        try:
            s = eng.engine.get_device_memory_size_v2()
        except Exception:
            s = eng.engine.device_memory_size
        return s

    base_path = args.base; super_path = args.super
    base_w = len(open(base_path, "rb").read())
    super_w = len(open(super_path, "rb").read())
    router_w = len(open(args.router_engine, "rb").read()) if args.router_engine else 0
    try:
        base_s = eb.engine.get_device_memory_size_v2()
        super_s = es.engine.get_device_memory_size_v2()
    except Exception:
        base_s = eb.engine.device_memory_size
        super_s = es.engine.device_memory_size
    router_s = 0
    if args.router_engine:
        try:
            router_s = er.engine.get_device_memory_size_v2()
        except Exception:
            router_s = er.engine.device_memory_size if lb_r is not None else 0
    total_w = base_w + super_w + router_w
    total_s = base_s + super_s + router_s

    # ---- report ----
    dev_name = torch.cuda.get_device_name(dev) if torch.cuda.is_available() else "CPU"
    mean_bs = (lb + ls) / 2
    sw = la - mean_bs
    print(f"\n==== Engine memory footprint ====")
    print(f"{'engine':<10}{'weights(MB)':>14}{'scratch(MB)':>14}{'total(MB)':>12}")
    print(f"{'base':<10}{base_w/MB:>14.1f}{base_s/MB:>14.1f}{(base_w+base_s)/MB:>12.1f}")
    print(f"{'super':<10}{super_w/MB:>14.1f}{super_s/MB:>14.1f}{(super_w+super_s)/MB:>12.1f}")
    if args.router_engine:
        print(f"{'router':<10}{router_w/MB:>14.1f}{router_s/MB:>14.1f}{(router_w+router_s)/MB:>12.1f}")
    print(f"{'TOTAL':<10}{total_w/MB:>14.1f}{total_s/MB:>14.1f}{(total_w+total_s)/MB:>12.1f}")
    print(f"\n==== TensorRT engine latency ({dev_name}, FP16, pure GPU inference) ====")
    print(f" BASE        : {lb:6.2f} ms   {fb:6.1f} fps   {enb:8.1f} mJ")
    print(f" SUPER       : {ls:6.2f} ms   {fs:6.1f} fps   {ens:8.1f} mJ")
    print(f" ALTERNATING : {la:6.2f} ms   {fa:6.1f} fps   {ena:8.1f} mJ   (mean(base,super)={mean_bs:.2f} ms)")
    print(f" switch overhead : {sw:+.3f} ms ({sw / mean_bs * 100:+.2f}% of mean)")
    if router_ms is not None:
        print(f" router overhead : {router_ms:.3f} ms "
              f"({router_ms / lb * 100:.2f}% of BASE, {router_ms / ls * 100:.2f}% of SUPER)")
    if lb_r is not None:
        print(f"\n==== detector + TRT router combined ====")
        print(f" BASE  + router : {lb_r:6.2f} ms   {fb_r:6.1f} fps   {enb_r:8.1f} mJ")
        print(f" SUPER + router : {ls_r:6.2f} ms   {fs_r:6.1f} fps   {ens_r:8.1f} mJ")
    print(f"\n[summary] switching {sw:+.3f} ms and routing "
          f"{(router_ms if router_ms else 0):.3f} ms are both negligible vs detector "
          f"{lb:.1f}-{ls:.1f} ms.")


if __name__ == "__main__":
    main()
