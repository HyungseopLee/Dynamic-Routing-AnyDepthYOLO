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
    """Wraps the detector to run ONE fixed depth path (skip baked in). Returns the
    standard inference-format detection output and -- so the router can run on-device --
    the pooled router feature (adaptive-avg-pool of INPUT_LEVEL_LAYERS to GxG, exactly
    matching grid_vec used during router training). Traces to a static ONNX graph."""

    def __init__(self, model, skip, tap_layers=(4, 6, 8, 14, 17, 20), grid=2):
        super().__init__()
        self.model = model
        self.skip = skip
        self.tap_layers = tap_layers
        self._cap = {}
        for l in tap_layers:
            model.model[l].register_forward_hook(
                lambda m, i, o, l=l: self._cap.__setitem__(l, o))
        self.model.model[-1].export = True
        self.model.model[-1].format = "onnx"

    def forward(self, x):
        # Output detection + the RAW INPUT_LEVEL feature maps. The (2x2) adaptive pool
        # that forms the router input is ONNX-unfriendly (non-factor output), so we do
        # it on the host with the exact training op -- keeping router input identical.
        self._cap.clear()
        det = self.model._predict_once(x, skip=self.skip, return_features=False)
        return (det, *[self._cap[l] for l in self.tap_layers])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--imgsz", type=int, nargs=2, required=True, help="H W")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--grid", type=int, default=2, help="router feature grid GxG (match policy)")
    ap.add_argument("--tap", type=int, nargs="+", default=[4, 6, 8, 14, 17, 20],
                    help="layers to emit as raw feature outputs (input-level 4,6,8 and, for "
                         "feat=both policies, pred-level 14,17,20)")
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

    feat_names = [f"feat{l}" for l in args.tap]
    for name, skip in [("base", [True] * N), ("super", [False] * N)]:
        wrapper = PathModel(m, skip, tap_layers=tuple(args.tap), grid=args.grid).eval()
        out = out_dir / f"{name}.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, str(out), opset_version=args.opset,
                input_names=["images"],
                output_names=["output", *feat_names],
                dynamic_axes=None)
        print(f"[*] {name}: skip={'all' if name=='base' else 'none'} -> {out}  "
              f"(input 1x3x{H}x{W}, +raw tap maps {args.tap})")


if __name__ == "__main__":
    main()
