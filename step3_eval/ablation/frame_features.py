"""Glimpse's frame-difference trigger, used here as a scene-change guard.

Glimpse [Chen et al., SenSys'15] detects scene changes with a two-stage frame
differencing rule (paper Section 3.2 / 4, Eq. 1-3):

  1. Convert consecutive frames to grayscale; for every pixel,
         a(x,y) = |f_t(x,y) - f_{t-1}(x,y)|
         d(x,y) = 1 if a(x,y) > phi else 0            phi = 35
     phi=35 is the paper's own empirically-chosen constant -- Section 3.2 gives no
     derivation for it beyond "based on our experiments, sensitive enough to capture
     scene movements and robust to changes caused by noise" -- so it is an intensity
     threshold (8-bit grayscale units), not a pixel-count, and transplants unchanged
     to any resolution.
  2. D_t = sum(d(x,y)) is the changed-pixel count. Glimpse triggers when
         D_t > phi_motion,   phi_motion = 0.5 * (pixels in a frame)
     phi_motion is stated as "half the number of pixels in a frame" (Section 7.3):
     a resolution-RELATIVE rule, so we compute it fresh for each dataset's
     resolution rather than reusing the paper's absolute number (153600, half of
     their 640x480). Storing the changed-pixel FRACTION (D_t / total pixels) makes
     this trivial: the trigger is simply fraction > 0.5, at any resolution.

This uses only raw pixels of the current frame, so a guard built on it may act on
frame t before any network runs -- unlike a guard built on router features, which
exist only after a path has been executed.

Written per sequence in evaluation order, so the array lines up index-for-index with
the frame dump produced by step3_eval/eval_video.py --frame_dump.

Usage (run from repo root):
    python -m step3_eval.ablation.frame_features --dataset kitti \
        --out results/step3_eval/kitti/eval/frame_features.npz
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from step3_eval.eval_video import _bdd_frames, _kitti_frames, _waymo_frames
from step3_eval.eval_video import _bdd_segments, _kitti_segments, _waymo_segments

FRAMES = {"kitti": _kitti_frames, "bdd100k": _bdd_frames, "waymo": _waymo_frames}
SEGS = {"kitti": _kitti_segments, "bdd100k": _bdd_segments, "waymo": _waymo_segments}
ROOTS = {"kitti": "/media/data/kitti-tracking",
         "bdd100k": "/media/data/bdd100k_mot/val",
         "waymo": "/media/data/waymo_yolo/val"}

PHI = 35   # Glimpse's per-pixel intensity-difference threshold (Eq. 1-2)


def glimpse_fraction(prev_gray, cur_gray):
    """Changed-pixel fraction between two grayscale frames (Glimpse Eq. 1-3)."""
    d = cv2.absdiff(cur_gray, prev_gray)
    d = cv2.threshold(d, PHI, 255, cv2.THRESH_BINARY)[1]
    return cv2.countNonZero(d) / d.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti", choices=list(FRAMES))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.data_root = args.data_root or ROOTS[args.dataset]
    args.sequences = None
    args.limit = 0
    args.shuffle_frames = False
    args.shuffle_seed = 0

    seqs = SEGS[args.dataset](args)
    names, vals = [], []
    cost = 0.0
    for i, seq in enumerate(seqs):
        prev, n = None, 0
        for _, bgr in FRAMES[args.dataset](seq, args):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev is None:
                vals.append(0.0)
            else:
                t0 = time.perf_counter()
                vals.append(glimpse_fraction(prev, gray))
                cost += time.perf_counter() - t0
            names.append(seq)
            prev, n = gray, n + 1
        print(f"[{i+1}/{len(seqs)} {seq}] {n} frames", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, seq=np.array(names),
                        glimpse=np.asarray(vals, np.float32))
    print(f"[*] {len(vals)} frames -> {out}")
    print(f"[*] cost/frame: {cost / max(len(vals) - len(seqs), 1) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
