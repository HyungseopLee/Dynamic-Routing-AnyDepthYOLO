"""
AnyDepth Step 3 — compute per-percentile router-score thresholds on a val set.

Ported from DynamicDet/get_dynamic_thres.py. The router's difficulty scores are
collected over the validation set; thresholds are the 0/10/.../90/100 percentiles.

Interpretation:
    With threshold = thres[k],  SUPER is used for (100-10*k)% of samples.

Usage:
    python get_dynamic_thres.py \
        --weight runs/bdd100k/step2/anydepth-yolov12l/step2_router/weights/last.pt \
        --data   bdd100k.yaml \
        --imgsz  1280 \
        --split  val
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, required=True, help='Step 2 checkpoint (with router)')
    p.add_argument('--data', type=str, required=True, help='Dataset yaml')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--half', action='store_true')
    p.add_argument('--out', type=str, default='',
                   help='Optional path to save thresholds (txt). If empty, only printed.')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # Load Step2 checkpoint
    model_wrap = YOLO(args.weight)
    model = model_wrap.model  # underlying nn.Module
    assert hasattr(model, 'router'), \
        "Checkpoint does not expose a `.router` — did you train Step 2 first?"
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    model.to(device).eval()
    if args.half and device.type == 'cuda':
        model.half()

    # Build val dataloader via ultralytics
    overrides = dict(
        model=args.weight,
        data=args.data,
        imgsz=args.imgsz,
        batch=1,
        rect=True,
        task='detect',
        mode='val',
    )
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(args.data)
    img_path = data_dict[args.split]
    stride = max(int(model.stride.max() if hasattr(model, 'stride') else 32), 32)
    dataset = build_yolo_dataset(cfg, img_path, batch=1, data=data_dict, mode='val',
                                 rect=True, stride=stride)
    loader = build_dataloader(dataset, batch=1, workers=4, shuffle=False, rank=-1)

    scores = []
    for batch in tqdm(loader, desc='Collecting router scores'):
        img = batch['img'].to(device, non_blocking=True)
        img = img.half() if args.half else img.float()
        img = img / 255.0
        score = model.predict(img, get_score=True)  # [1, 1] sigmoided
        scores.append(float(score.view(-1)[0]))

    scores = np.array(scores)
    print(f"\n[Step3] collected {len(scores)} scores  "
          f"min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}")

    thres = ['0.0']
    for p in range(10, 100, 10):
        thres.append(f"{np.percentile(scores, p):.6f}")
    thres.append('1.0')

    print()
    print('*' * 72)
    print(' '.join(thres))
    for idx, thr in enumerate(thres):
        print(f"  SUPER: {100 - idx * 10:>3}%   BASE: {idx * 10:>3}%   threshold: {thr}")
    print('*' * 72)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(' '.join(thres) + '\n')
        print(f"[Step3] thresholds written to {out_path}")


if __name__ == '__main__':
    main()


'''
# Example — BDD100K
python get_dynamic_thres.py \
    --weight runs/bdd100k/step2/anydepth-yolov12l/step2_router/weights/last.pt \
    --data   bdd100k.yaml \
    --imgsz  1280 \
    --split  val \
    --device 0 \
    --out    runs/bdd100k/step2/anydepth-yolov12l/step2_router/thres.txt
'''
