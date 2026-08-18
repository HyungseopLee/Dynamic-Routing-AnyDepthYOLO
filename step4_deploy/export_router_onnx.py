"""Export the advantage-regression router (RouterNetwork) as a standalone ONNX graph
for TensorRT, so routing runs on-device in TRT instead of eager PyTorch.

The deployed router is a SINGLE fixed graph: the path_embed (path-conditioning) is a
training-time technique, and for this checkpoint its effect is negligible anyway
(||emb[base]-emb[super]|| ~ 0.1, logit shift ~ 0.015), so we bake one fixed path id.

Inputs are the pre-pooled (C,grid,grid) tap groups the detector engine already emits:
  feat=both  -> iv = cat(input-level taps), pv = cat(pred-level taps)  -> logit
  feat=input -> iv only                                               -> logit

Requires a pooled detector engine (export_onnx.py --pool) to read iv/pv channel dims.
After export, build the TRT engine with build_engine.py and pass it via --router_engine
to online_budget_demo_stream.py.

    python -m step4_deploy.export_router_onnx \
        --router results/step2_router/weights/bdd100k/router_both_0.pt \
        --base_engine results/step4_deploy/onnx/bdd_pooled/base.fp16.engine \
        --out results/step4_deploy/onnx/bdd_pooled/router.onnx

    python -m step4_deploy.build_engine \
        --onnx results/step4_deploy/onnx/bdd_pooled/router.onnx --fp16
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from step3_eval.eval_video import load_router
from router.feature_tap import INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS


def tap_channels(engine_path, layers):
    import tensorrt as trt
    eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
        open(engine_path, "rb").read())
    name2shape = {eng.get_tensor_name(i): tuple(eng.get_tensor_shape(eng.get_tensor_name(i)))
                  for i in range(eng.num_io_tensors)}
    grid = name2shape[f"feat{layers[0]}"][-1]
    return sum(name2shape[f"feat{l}"][1] for l in layers), grid


class RouterWrap(nn.Module):
    """net.logit(iv, pv, fixed_pid) as a 2-input (or 1-input) graph -> logit."""

    def __init__(self, net, use_pred, pid):
        super().__init__()
        self.net = net
        self.use_pred = use_pred
        self.register_buffer("pid", torch.tensor([pid], dtype=torch.long))

    def forward(self, iv, pv=None):
        return self.net.logit(iv, pv if self.use_pred else None, self.pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--base_engine", required=True,
                    help="a pooled detector engine, to read iv/pv channel dims + grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pid", type=int, default=1, help="fixed path id to bake (0=base,1=super)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    net, ckpt, feat, is_gap = load_router(args.router, dev)
    use_pred = feat in ("both", "pred")
    in_c, grid = tap_channels(args.base_engine, INPUT_LEVEL_LAYERS)
    pr_c = tap_channels(args.base_engine, PRED_LEVEL_LAYERS)[0] if use_pred else 0
    # materialize the lazy net at the real tap dims, then load weights
    with torch.no_grad():
        net(torch.zeros(2, in_c, grid, grid, device=dev),
            torch.zeros(2, pr_c, grid, grid, device=dev) if use_pred else None,
            torch.zeros(2, dtype=torch.long, device=dev))
    net.load_state_dict(ckpt["state_dict"]); net.eval()

    wrap = RouterWrap(net, use_pred, args.pid).to(dev).eval()
    iv = torch.zeros(1, in_c, grid, grid, device=dev)
    pv = torch.zeros(1, pr_c, grid, grid, device=dev) if use_pred else None
    inputs = (iv, pv) if use_pred else (iv,)
    in_names = ["iv", "pv"] if use_pred else ["iv"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrap, inputs, args.out, opset_version=args.opset,
                          input_names=in_names, output_names=["logit"], dynamic_axes=None)
    print(f"[*] router ({feat}, pid={args.pid}) -> {args.out}  "
          f"iv=({in_c},{grid},{grid})" + (f" pv=({pr_c},{grid},{grid})" if use_pred else ""))


if __name__ == "__main__":
    main()
