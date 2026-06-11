"""Pick samples where the SUPER path is clearly better than BASE and visualize.

Uses the cached per-image losses to rank validation images by advantage
A = L_base - L_super (large A = SUPER helps most), then runs both paths on the
top-K images and saves, per image, three PNGs: raw, BASE detections, SUPER
detections. Run in the yolov12 conda env.

    conda run -n yolov12 python paper/make_compare_samples.py
"""

import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent / "anydepth-yolov12"
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402

WEIGHT = ROOT / "runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt"
CACHE = ROOT / "method02_advantage_regress_tinyConv/outputs/kitti/cache_val_g2.pt"
OUT = Path(__file__).resolve().parent / "_extracted" / "samples"
CAND, SAVEK, IMGSZ, CONF = 300, 8, (384, 1248), 0.30   # scan CAND high-A imgs, save best SAVEK


def count(yolo, bgr, skip):
    r = yolo.predict(source=bgr, imgsz=IMGSZ, conf=CONF, iou=0.7, skip=skip, verbose=False)[0]
    n = 0 if r.boxes is None else len(r.boxes)
    return n, r


def draw(img, r, color, thick=2):
    """Draw a result's boxes on img in a fixed BGR color."""
    if r.boxes is None:
        return img
    for b in r.boxes.xyxy.cpu().numpy().astype(int):
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, thick)
    return img


def load_gt(img_path, h, w):
    """YOLO-format label file -> list of pixel (x1,y1,x2,y2)."""
    lp = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    boxes = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, bw, bh = (float(x) for x in p[1:5])
            boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                          int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return boxes


def legend(img):
    """Top-left colour key: GT green, base blue, super red."""
    items = [("GT", (0, 200, 0)), ("base", (255, 0, 0)), ("super", (0, 0, 255))]
    x, y = 8, 8
    for i, (txt, col) in enumerate(items):
        yy = y + i * 26
        cv2.rectangle(img, (x, yy), (x + 22, yy + 18), col, -1)
        cv2.putText(img, txt, (x + 28, yy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img


def overlay(bgr, r_base, r_super, img_path):
    """One frame with GT (green), BASE (blue), SUPER (red)."""
    img = bgr.copy()
    h, w = img.shape[:2]
    for b in load_gt(img_path, h, w):
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 200, 0), 2)  # GT green
    draw(img, r_base, (255, 0, 0))      # blue
    draw(img, r_super, (0, 0, 255))     # red
    return legend(img)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = torch.load(CACHE, map_location="cpu", weights_only=False)
    A = (c["loss_base"] - c["loss_super"]).view(-1)
    cand = torch.argsort(A, descending=True)[:CAND].tolist()

    yolo = YOLO(str(WEIGHT), task="detect")
    nskip = getattr(yolo.model, "num_skippable_layers", 0)
    sk = {"base": [True] * nskip, "super": [False] * nskip}

    # rank candidates by how many MORE boxes SUPER finds than BASE (recall gap)
    scored = []
    for i in cand:
        fp = Path(c["im_file"][i])
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        nb, _ = count(yolo, bgr, sk["base"])
        ns, _ = count(yolo, bgr, sk["super"])
        scored.append((ns - nb, float(A[i]), i, fp))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    for rank, (diff, a, i, fp) in enumerate(scored[:SAVEK]):
        bgr = cv2.imread(str(fp))
        stem = f"r{rank:02d}_{fp.stem}_A{a:.2f}_d{diff:+d}"
        cv2.imwrite(str(OUT / f"{stem}_raw.png"), bgr)
        res = {}
        for tag in ("base", "super"):
            _, res[tag] = count(yolo, bgr, sk[tag])
            cv2.imwrite(str(OUT / f"{stem}_{tag}.png"), res[tag].plot())
        # combined: BASE blue + SUPER red on one frame
        cv2.imwrite(str(OUT / f"{stem}_overlay.png"), overlay(bgr, res["base"], res["super"], fp))
        print(f"[*] {stem}: base->super box diff {diff:+d}")


if __name__ == "__main__":
    main()
