"""Benchmark depth routing under TensorRT: load the BASE and SUPER engines, hold
both resident in VRAM, and measure pure GPU inference latency for

  * BASE only       (always-base anchor),
  * SUPER only      (always-super anchor),
  * ALTERNATING     (worst-case engine switching every frame),

so the per-frame engine-switch overhead is isolated (alternating vs the mean of the
two single-path runs). GPU timing uses CUDA events; energy via pynvml. I/O buffers are
torch tensors kept resident -- engines are loaded ONCE, never reloaded per frame.

    python method_advantage_regress/jetson/bench_trt.py --base method_advantage_regress/jetson/onnx/bdd/base.fp16.engine \
        --super method_advantage_regress/jetson/onnx/bdd/super.fp16.engine --iters 1000 --warmup 100
"""
import argparse
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class Engine:
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


class Power:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit(); self.h = pynvml.nvmlDeviceGetHandleByIndex(0); self.p = pynvml
        except Exception:
            self.p = None

    def mw(self):
        return float(self.p.nvmlDeviceGetPowerUsage(self.h)) if self.p else 0.0


def timed(run_fn, iters, warmup, power):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            run_fn(stream)
    torch.cuda.synchronize()
    lat, eng = [], []
    with torch.cuda.stream(stream):
        for i in range(iters):
            p0 = power.mw()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(stream); run_fn(stream); e.record(stream)
            e.synchronize()
            ms = s.elapsed_time(e)
            p1 = power.mw()
            lat.append(ms); eng.append(0.5 * (p0 + p1) * ms / 1000.0)
    return float(np.mean(lat)), 1000.0 / float(np.mean(lat)), float(np.mean(eng))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--super", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    power = Power()

    eb = Engine(args.base, dev); es = Engine(args.super, dev)
    print(f"[*] base in  {tuple(eb.buffers[eb.inp].shape)} | super in {tuple(es.buffers[es.inp].shape)}")
    # memory footprint of holding BOTH engines resident -- the key constraint on
    # memory-limited devices (e.g. Jetson). device_memory_size is the per-context
    # activation scratch; weights + I/O buffers add on top (reported via torch).
    mb = lambda b: b / (1024 ** 2)
    dm = mb(eb.engine.device_memory_size + es.engine.device_memory_size)
    print(f"[*] resident GPU memory: two-engine activation scratch {dm:.0f} MB; "
          f"torch reserved {mb(torch.cuda.memory_reserved(dev)):.0f} MB "
          f"(allocated {mb(torch.cuda.memory_allocated(dev)):.0f} MB)")

    lb, fb, enb = timed(lambda s: eb.run(s), args.iters, args.warmup, power)
    ls, fs, ens = timed(lambda s: es.run(s), args.iters, args.warmup, power)

    flip = {"v": True}
    def alt(s):
        flip["v"] = not flip["v"]
        (eb if flip["v"] else es).run(s)
    la, fa, ena = timed(alt, args.iters, args.warmup, power)

    dev_name = torch.cuda.get_device_name(dev) if torch.cuda.is_available() else "CPU"
    print(f"\n==== TensorRT engine latency ({dev_name}, pure GPU inference) ====")
    print(f" BASE       : {lb:6.2f} ms   {fb:6.1f} fps   {enb:8.1f} mJ")
    print(f" SUPER      : {ls:6.2f} ms   {fs:6.1f} fps   {ens:8.1f} mJ")
    print(f" ALTERNATING: {la:6.2f} ms   {fa:6.1f} fps   {ena:8.1f} mJ   (mean(base,super)={ (lb+ls)/2:.2f} ms)")
    print(f" switch overhead vs mean: {la - (lb+ls)/2:+.3f} ms ({(la-(lb+ls)/2)/((lb+ls)/2)*100:+.2f}%)")


if __name__ == "__main__":
    main()
