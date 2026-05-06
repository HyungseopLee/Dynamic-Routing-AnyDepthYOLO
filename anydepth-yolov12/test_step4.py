"""
AnyDepth Step 4 — evaluate the router + threshold pair(s) on a validation set.

For each threshold T_k produced by Step 3:
  * per-image: if router_score >= T_k  -> SUPER (skip=[False]*N)
               else                     -> BASE  (skip=[True] *N)
  * full detection mAP is reported together with (super_ratio, base_ratio).

Usage:
    python test_step4.py \
        --weight runs/bdd100k/step2/anydepth-yolov12l/step2_router/weights/last.pt \
        --data   bdd100k.yaml \
        --imgsz  1280 \
        --thres-file runs/bdd100k/step2/anydepth-yolov12l/step2_router/thres.txt
"""
import argparse
from copy import copy
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, required=True, help='Step 2 checkpoint (with router)')
    p.add_argument('--data', type=str, required=True, help='Dataset yaml')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--thres-file', type=str, default='',
                   help='Text file (one line, space-separated) from get_dynamic_thres.py')
    p.add_argument('--thres', type=str, default='',
                   help='Comma-separated list of thresholds, overrides --thres-file')
    p.add_argument('--project', type=str, default='runs/step4')
    p.add_argument('--name', type=str, default='routed')
    return p.parse_args()


def _parse_thres(args):
    if args.thres:
        return [float(x) for x in args.thres.split(',') if x.strip()]
    if args.thres_file:
        with open(args.thres_file) as f:
            return [float(x) for x in f.read().strip().split()]
    raise ValueError("Provide --thres-file or --thres")


def run_one_threshold(model_wrap, data, imgsz, dy_thres, project, name, device):
    """Run validation with per-image SUPER/BASE routing at a fixed threshold."""
    model = unwrap_model(model_wrap.model)
    assert hasattr(model, 'router'), "checkpoint has no router"

    # Attach thres via attribute — picked up by DetectionModelAnyDepth.predict
    # when dataloader feeds bs=1 inference.
    model._fixed_dy_thres = dy_thres

    # Monkey-patch predict to pass dy_thres automatically
    orig_predict = model.predict
    def _predict_with_thres(x, **kwargs):
        kwargs.setdefault('dy_thres', model._fixed_dy_thres)
        return orig_predict(x, **kwargs)
    model.predict = _predict_with_thres

    # Also count super/base usage
    super_count = {'n': 0}
    base_count = {'n': 0}
    orig_dynamic = model._predict_dynamic
    def _counting_dynamic(x, dy_thres, return_score=False):
        # run once, count, and return the detection output
        tap = model._forward_shared_tap(x)
        score = model.router(tap)
        B = x.shape[0]
        assert B == 1, "batch_size=1 required for routed inference"
        use_super = score.view(-1)[0].item() >= dy_thres
        if use_super:
            super_count['n'] += 1
            skip = [False] * model.num_skippable_layers
        else:
            base_count['n'] += 1
            skip = [True] * model.num_skippable_layers
        out = model._predict_once(x, False, False, None, skip=skip, return_features=False)
        return (out, score) if return_score else out
    model._predict_dynamic = _counting_dynamic

    try:
        results = model_wrap.val(
            data=data,
            imgsz=imgsz,
            batch=1,
            device=device,
            project=project,
            name=f'{name}_T{dy_thres:.4f}',
            save_json=False,
            rect=True,
        )
    finally:
        # restore
        model.predict = orig_predict
        model._predict_dynamic = orig_dynamic
        del model._fixed_dy_thres

    total = super_count['n'] + base_count['n']
    ratio_super = super_count['n'] / max(total, 1)
    ratio_base = base_count['n'] / max(total, 1)
    return results, ratio_super, ratio_base


def main():
    args = parse_args()
    thresholds = _parse_thres(args)

    LOGGER.info(f"[Step4] evaluating {len(thresholds)} threshold(s): {thresholds}")
    model_wrap = YOLO(args.weight)

    summary = []  # (thres, mAP50-95, super_ratio)
    for T in thresholds:
        LOGGER.info(f"\n{'='*72}\n[Step4] threshold = {T}\n{'='*72}")
        results, rs, rb = run_one_threshold(
            model_wrap, args.data, args.imgsz, T,
            args.project, args.name, args.device,
        )
        map5095 = float(results.box.map) if hasattr(results, 'box') else float('nan')
        LOGGER.info(
            f"[Step4] T={T:.4f}  mAP50-95={map5095:.4f}  "
            f"SUPER={rs*100:.1f}%  BASE={rb*100:.1f}%"
        )
        summary.append((T, map5095, rs, rb))

    LOGGER.info(f"\n{'='*72}\n[Step4] SUMMARY\n{'='*72}")
    LOGGER.info(f"{'threshold':>12} {'mAP50-95':>10} {'SUPER%':>8} {'BASE%':>8}")
    for T, m, rs, rb in summary:
        LOGGER.info(f"{T:>12.4f} {m:>10.4f} {rs*100:>7.1f}% {rb*100:>7.1f}%")


if __name__ == '__main__':
    main()


'''
# Example — BDD100K
python test_step4.py \
    --weight runs/bdd100k/step2/anydepth-yolov12l/step2_router/weights/last.pt \
    --data   bdd100k.yaml \
    --imgsz  1280 \
    --thres-file runs/bdd100k/step2/anydepth-yolov12l/step2_router/thres.txt \
    --project runs/bdd100k/step4 \
    --name anydepth-yolov12l
'''
