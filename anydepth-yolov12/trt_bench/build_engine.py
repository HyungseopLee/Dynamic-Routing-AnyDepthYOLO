"""Build a TensorRT engine from an ONNX file (FP16 by default).

    python trt_bench/build_engine.py --onnx trt_bench/onnx/bdd/base.onnx --fp16
"""
import argparse
from pathlib import Path

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build(onnx_path, engine_path, fp16=True, workspace_gb=8):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"failed to parse {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[*] {onnx_path} -> {engine_path}  (fp16={fp16})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="force FP32 (overrides --fp16)")
    args = ap.parse_args()
    engine = args.engine or str(Path(args.onnx).with_suffix(
        ".fp32.engine" if args.fp32 else ".fp16.engine"))
    build(args.onnx, engine, fp16=not args.fp32)


if __name__ == "__main__":
    main()
