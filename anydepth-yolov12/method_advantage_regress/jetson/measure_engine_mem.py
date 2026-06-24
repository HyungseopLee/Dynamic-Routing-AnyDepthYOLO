"""Report the resident GPU-memory footprint of each TRT engine: the serialized
weights (engine size) and the activation/scratch device memory the execution
context needs. This is the memory that must stay resident to keep BASE, SUPER and
the router co-resident so engine switching is a pointer flip, not a reload."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorrt as trt

L = trt.Logger(trt.Logger.ERROR)


def info(path):
    blob = open(path, "rb").read()
    eng = trt.Runtime(L).deserialize_cuda_engine(blob)
    ctx = eng.create_execution_context()
    try:
        scratch = eng.get_device_memory_size_v2()
    except Exception:
        scratch = eng.device_memory_size
    return len(blob), scratch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", required=True,
                    help="name=path pairs, e.g. base=method_advantage_regress/jetson/onnx/bdd_pooled/base.fp16.engine")
    args = ap.parse_args()
    MB = 1024 ** 2
    tot_w = tot_s = 0
    print(f"{'engine':<10}{'weights(MB)':>14}{'scratch(MB)':>14}{'total(MB)':>12}")
    for spec in args.engines:
        name, path = spec.split("=", 1)
        w, s = info(path)
        tot_w += w; tot_s += s
        print(f"{name:<10}{w/MB:>14.1f}{s/MB:>14.1f}{(w+s)/MB:>12.1f}")
    print(f"{'TOTAL':<10}{tot_w/MB:>14.1f}{tot_s/MB:>14.1f}{(tot_w+tot_s)/MB:>12.1f}")


if __name__ == "__main__":
    main()
