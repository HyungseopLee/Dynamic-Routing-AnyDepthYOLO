"""
AnyDepth Step 3 (KITTI variant) — collect router scores over KITTI tracking
sequences and emit percentile thresholds.

Use this when you want a KITTI-native threshold for in-domain evaluation
(i.e. scenario "(C) 둘 다 비교" alongside thres derived from BDD100K val).

Usage:
    python get_dynamic_thres_kitti.py \
        --weight runs/bdd100k/step2/.../step2_router/weights/last.pt \
        --kitti_root /media/data/kitti-tracking \
        --sequences 0000 0001 0002 0003 \
        --imgsz 192 640 \
        --out runs/.../thres_kitti.txt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

from eval_video_kitti_dynamic import _preprocess_for_router


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', required=True)
    p.add_argument('--kitti_root', default='/media/data/kitti-tracking')
    p.add_argument('--sequences', nargs='+', default=None)
    p.add_argument('--imgsz', type=int, nargs='+', default=[192, 640])
    p.add_argument('--device', default='0')
    p.add_argument('--out', type=str, default='', help='Save threshold file')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
    else:
        imgsz = tuple(args.imgsz)

    model_wrap = YOLO(args.weight)
    inner = unwrap_model(model_wrap.model)
    assert hasattr(inner, 'router'), "checkpoint has no router"
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    inner.to(device).eval()
    stride = int(getattr(inner, 'stride', torch.tensor([32])).max())

    if args.sequences is None:
        args.sequences = sorted([f.stem for f in (Path(args.kitti_root) / 'training' / 'label_02').glob('*.txt')])

    scores = []
    for seq_id in args.sequences:
        img_dir = Path(args.kitti_root) / 'training' / 'image_02' / seq_id
        frames = sorted(img_dir.glob('*.png')) or sorted(img_dir.glob('*.jpg'))
        for fp in tqdm(frames, desc=f'seq {seq_id}'):
            frame = cv2.imread(str(fp))
            t = _preprocess_for_router(frame, imgsz, stride=stride, device=device, half=False)
            s = inner.predict(t, get_score=True)
            scores.append(float(s.view(-1)[0]))

    scores = np.asarray(scores)
    print(f"\n[*] {len(scores)} scores  min={scores.min():.4f}  "
          f"max={scores.max():.4f}  mean={scores.mean():.4f}")

    thres = ['0.0']
    for p in range(10, 100, 10):
        thres.append(f"{np.percentile(scores, p):.6f}")
    thres.append('1.0')

    print('\n' + '*' * 72)
    print(' '.join(thres))
    for i, t in enumerate(thres):
        print(f"  SUPER: {100 - i * 10:>3}%   BASE: {i * 10:>3}%   T: {t}")
    print('*' * 72)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(' '.join(thres) + '\n')
        print(f"\n[*] saved to {args.out}")


if __name__ == '__main__':
    main()
