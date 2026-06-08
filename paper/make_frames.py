"""Save illustration frames for the approach figure.

For a short clip (every STRIDE-th frame of a KITTI-tracking sequence) save, per
frame, two PNGs: the raw frame and the frame with SUPER-path detection results
(boxes + class labels) drawn. Run in the yolov12 conda env (torchvision needed).

    conda run -n yolov12 python paper/make_frames.py
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent / "anydepth-yolov12"
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402

WEIGHT = ROOT / "runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt"
SEQ = "0001"                        # higher-motion sequence
SEQ_DIR = Path(f"/media/data/kitti-tracking/training/image_02/{SEQ}")
OUT = Path(__file__).resolve().parent / "_extracted" / "frames"
START, STRIDE, N = 0, 1, 21         # frames 0..20 (all)
IMGSZ, CONF = (384, 1248), 0.40


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(str(WEIGHT), task="detect")
    nskip = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super = [False] * nskip

    frames = sorted(SEQ_DIR.glob("*.png"))
    picks = [frames[START + i * STRIDE] for i in range(N) if START + i * STRIDE < len(frames)]
    for fp in picks:
        bgr = cv2.imread(str(fp))
        raw_out = OUT / f"seq{SEQ}_{fp.stem}_raw.png"
        cv2.imwrite(str(raw_out), bgr)
        r = yolo.predict(source=bgr, imgsz=IMGSZ, conf=CONF, iou=0.7,
                         skip=skip_super, verbose=False)[0]
        det = r.plot()                      # BGR with boxes + class labels
        det_out = OUT / f"seq{SEQ}_{fp.stem}_det.png"
        cv2.imwrite(str(det_out), det)
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"[*] {fp.stem}: {n} boxes -> {raw_out.name}, {det_out.name}")


if __name__ == "__main__":
    main()
