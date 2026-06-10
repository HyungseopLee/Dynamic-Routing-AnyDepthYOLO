"""Save illustration frames for the approach figure — BDD100K MOT version.

Like make_frames.py (KITTI) but the frame source is a decoded box_track_20 `.mov`
clip instead of a PNG directory, the detector is the BDD-finetuned AnyDepth weight,
and clips may carry portrait rotation metadata that we straighten. For each picked
frame we save two PNGs: the raw frame and the frame with SUPER-path detections.

    conda run -n yolov12 python paper/make_frames_bdd.py [SEQ_STEM]
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent / "anydepth-yolov12"
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402
from method02_advantage_regress_tinyConv.eval_video_bdd import open_upright, decode_upright  # noqa: E402

WEIGHT = ROOT / "finetuned_bdd100k" / \
    "30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.2_orig_mAP35.1_33.8.pt"
VIDEO_DIR = Path("/media/data/bdd100k_mot/val/videos")
SEQ = sys.argv[1] if len(sys.argv) > 1 else "b1c66a42-6f7d68ca"   # a val clip
OUT = Path(__file__).resolve().parent / "_extracted" / "frames_bdd"
START, STRIDE, N = 0, 6, 21         # ~every 6th decoded frame (≈5 fps) , 21 picks
IMGSZ, CONF = (720, 1280), 0.40     # H, W ; conf high for a clean illustration


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(str(WEIGHT), task="detect")
    nskip = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super = [False] * nskip

    cap, meta = open_upright(VIDEO_DIR / f"{SEQ}.mov")
    if not cap.isOpened():
        raise SystemExit(f"cannot open {SEQ}.mov")
    picks = {START + i * STRIDE for i in range(N)}
    fi = saved = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fi in picks:
            bgr = decode_upright(bgr, meta)                        # version-robust straighten
            raw_out = OUT / f"{SEQ}_f{fi:04d}_raw.png"
            cv2.imwrite(str(raw_out), bgr)
            r = yolo.predict(source=bgr, imgsz=IMGSZ, conf=CONF, iou=0.7,
                             skip=skip_super, verbose=False)[0]
            det = r.plot()                  # BGR with boxes + class labels
            det_out = OUT / f"{SEQ}_f{fi:04d}_det.png"
            cv2.imwrite(str(det_out), det)
            n = 0 if r.boxes is None else len(r.boxes)
            print(f"[*] f{fi:04d}: {n} boxes -> {raw_out.name}, {det_out.name}")
            saved += 1
        fi += 1
        if saved >= N:
            break
    cap.release()
    print(f"[*] saved {saved} frame pairs -> {OUT}")


if __name__ == "__main__":
    main()
