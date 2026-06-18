"""Export the BASE and SUPER execution paths of a frozen AnyDepth-YOLOv12s detector
as two SEPARATE ONNX graphs (the depth choice is baked in via a fixed skip pattern).

These two static graphs are then converted to TensorRT engines (build_engine.py); at
runtime the router selects which engine to run per frame -- this is how depth routing
is realized under TensorRT, which cannot skip layers inside a single static engine.

    python trt_bench/export_onnx.py --weight <anydepth.pt> --imgsz 720 1280 --out_dir trt_bench/onnx/bdd
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from ultralytics import YOLO


class PathModel(nn.Module):
    """Wraps the detector to run ONE fixed depth path (skip baked in) and return the
    standard inference-format detection output, so it traces to a static ONNX graph."""

    def __init__(self, model, skip):
        super().__init__()
        self.model = model
        self.skip = skip
        # export-mode Detect head: emit the concatenated pre-NMS prediction tensor.
        self.model.model[-1].export = True
        self.model.model[-1].format = "onnx"

    def forward(self, x):
        return self.model._predict_once(x, skip=self.skip, return_features=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--imgsz", type=int, nargs=2, required=True, help="H W")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.weight, task="detect")
    m = yolo.model.to(dev).eval()
    if not hasattr(m, "num_skippable_layers"):
        from method02_advantage_regress_tinyConv.build_cache import num_skippable
        m.num_skippable_layers = num_skippable(m)
    N = m.num_skippable_layers
    H = (args.imgsz[0] + 31) // 32 * 32
    W = (args.imgsz[1] + 31) // 32 * 32
    dummy = torch.zeros(1, 3, H, W, device=dev)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for name, skip in [("base", [True] * N), ("super", [False] * N)]:
        wrapper = PathModel(m, skip).eval()
        out = out_dir / f"{name}.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, str(out), opset_version=args.opset,
                input_names=["images"], output_names=["output"],
                dynamic_axes=None)
        print(f"[*] {name}: skip={'all' if name=='base' else 'none'} -> {out}  (input 1x3x{H}x{W})")


if __name__ == "__main__":
    main()
