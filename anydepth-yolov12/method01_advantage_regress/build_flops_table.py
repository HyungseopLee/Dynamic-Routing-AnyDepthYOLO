"""Offline FLOPs lookup table for the 2-level AnyDepth policy.

Measures GFLOPs for the two action paths of the AnyDepth-YOLOv12 model:
  - SUPER : all switchable/skippable layers run the full path   (skip=[False]*N)
  - BASE  : all switchable/skippable layers run the essential path (skip=[True]*N)

The result is cached to a JSON file and consumed at train time by the
L_flops term (constant per action, no gradient).

Usage:
    
    python ./method01_advantage_regress/build_flops_table.py \
        --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
        --imgsz 384 1280 \
        --out runs/kitti/policy/flops_table.json
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

# Ensure the local (anydepth-yolov12) ultralytics is imported, not an installed one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from ultralytics import YOLO


def measure_gflops(model, skip_pattern, imgsz, device):
    """GFLOPs for a fixed skip pattern via thop, wrapping _predict_once as forward(x)."""
    import thop

    H, W = int(imgsz[0]), int(imgsz[1])
    H = ((H + 31) // 32) * 32
    W = ((W + 31) // 32) * 32
    x = torch.empty((1, 3, H, W), device=device)

    class _Wrap(nn.Module):
        def __init__(self, m, skip):
            super().__init__()
            self.m = m
            self.skip = skip

        def forward(self, x):
            return self.m._predict_once(x, False, False, None,
                                        skip=self.skip, return_features=False)

    wrapped = _Wrap(deepcopy(model).eval(), skip_pattern).to(device)
    macs, _ = thop.profile(wrapped, inputs=(x,), verbose=False)
    return float(macs) * 2 / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1280], help="H W")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="kitti", help="output scope: outputs/<dataset>/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(Path(__file__).resolve().parent / "outputs" / args.dataset / "flops_table.json")

    device = args.device if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.weight, task="detect")
    model = yolo.model.to(device).eval()

    num_skip = getattr(model, "num_skippable_layers", None)
    if not num_skip:
        # populate via a dummy predict if attribute not set yet
        model.predict = model.predict  # no-op; attribute set lazily in predict()
        from ultralytics.nn.tasks import (
            SkippableC3k2, SwitchableC3k2, SwitchableC2f, SwitchableA2C2f, SkippableA2C2f,
        )
        num_skip = sum(
            isinstance(m, (SkippableC3k2, SwitchableC3k2, SwitchableC2f, SwitchableA2C2f, SkippableA2C2f))
            for m in model.model
        )
        model.num_skippable_layers = num_skip

    skip_super = [False] * num_skip
    skip_base = [True] * num_skip

    with torch.no_grad():
        g_super = measure_gflops(model, skip_super, args.imgsz, device)
        g_base = measure_gflops(model, skip_base, args.imgsz, device)

    table = {
        "weight": str(args.weight),
        "imgsz": [int(args.imgsz[0]), int(args.imgsz[1])],
        "num_skippable_layers": int(num_skip),
        "actions": {
            "0_base": {"skip": skip_base, "gflops": round(g_base, 4)},
            "1_super": {"skip": skip_super, "gflops": round(g_super, 4)},
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2))
    print(f"[*] num_skippable_layers={num_skip}, imgsz={args.imgsz}")
    print(f"[*] BASE  GFLOPs = {g_base:.4f}")
    print(f"[*] SUPER GFLOPs = {g_super:.4f}")
    print(f"[*] saved -> {out}")


if __name__ == "__main__":
    main()
