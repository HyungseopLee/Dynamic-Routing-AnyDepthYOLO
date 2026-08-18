"""Build TRT engines from ONNX for RTX 3090 (FP16 by default, FP32 with --no-fp16).

Usage:
    python -m step4_deploy.build_engines_3090 --dataset bdd100k
    python -m step4_deploy.build_engines_3090 --dataset kitti
    python -m step4_deploy.build_engines_3090 --dataset waymo
    python -m step4_deploy.build_engines_3090  # all
    python -m step4_deploy.build_engines_3090 --no-fp16  # FP32
"""
import sys, os, ctypes
from pathlib import Path

_SP = Path(sys.executable).parent.parent / "lib/python3.11/site-packages"
_TRT_LIBS = _SP / "tensorrt_libs"
if str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))

# preload shared libs so tensorrt_bindings can find them
for lib in ["libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"]:
    p = _TRT_LIBS / lib
    if p.exists():
        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)

import tensorrt as trt
import argparse

ONNX_DIRS = {
    "bdd100k": "results/step4_deploy/onnx/bdd_pooled",
    "kitti":   "results/step4_deploy/onnx/kitti",
    "waymo":   "results/step4_deploy/onnx/waymo",
}

ENGINE_DIRS = {
    "bdd100k": "results/step4_deploy/onnx/bdd_pooled_3090",
    "kitti":   "results/step4_deploy/onnx/kitti_3090",
    "waymo":   "results/step4_deploy/onnx/waymo_3090",
}


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True,
                 fp32_shuffle: bool = False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [!] {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Workaround for cuTensor permutate bug on sm_86 (RTX 3090) at large feature maps:
    # force Shuffle layers (transpose/reshape ops used in attention) to FP32.
    if fp16 and fp32_shuffle:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        n_forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type == trt.LayerType.SHUFFLE:
                layer.precision = trt.DataType.FLOAT
                layer.set_output_type(0, trt.DataType.FLOAT)
                n_forced += 1
        print(f"  [mixed] forced {n_forced} Shuffle layers to FP32", flush=True)

    print(f"  Building {engine_path.name} ...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Failed to build engine for {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"  [*] -> {engine_path}  ({engine_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(ONNX_DIRS), default=None)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    ap.add_argument("--fp32-shuffle", action="store_true",
                    help="force Shuffle layers to FP32 (workaround for cuTensor sm_86 bug)")
    args = ap.parse_args()

    prec = "fp16" if args.fp16 else "fp32"
    if args.fp16 and args.fp32_shuffle:
        prec = "fp16_shfl32"
    datasets = [args.dataset] if args.dataset else list(ONNX_DIRS.keys())

    for ds in datasets:
        onnx_dir = Path(ONNX_DIRS[ds])
        eng_dir  = Path(ENGINE_DIRS[ds])
        if not onnx_dir.exists():
            print(f"[skip] ONNX dir not found: {onnx_dir}")
            continue
        print(f"\n[{ds}] {onnx_dir} -> {eng_dir}  ({prec})")
        for name in ["base", "super", "router"]:
            onnx_path = onnx_dir / f"{name}.onnx"
            if not onnx_path.exists():
                print(f"  [skip] {onnx_path} not found")
                continue
            engine_path = eng_dir / f"{name}.{prec}.engine"
            build_engine(onnx_path, engine_path, fp16=args.fp16,
                         fp32_shuffle=args.fp32_shuffle)


if __name__ == "__main__":
    main()
