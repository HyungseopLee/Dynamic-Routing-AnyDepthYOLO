"""Image-domain AP under the *exact same protocol* as eval_video_bdd.py, so the
det_20 image number is apples-to-apples with the MOT video always_super number.

Same knobs as the video eval: SUPER path (no skip), conf/iou, imgsz, the identical
8 movable classes (BDD_MOT_EVAL_CLS, dropping traffic light/sign), and the same
matcher / AP code (B.dataset_map_multi_iou). The only thing left differing from the
video eval is the *data domain* (det_20 keyframes vs MOT 5fps frames) -- so the gap
that remains is the genuine domain shift.

    conda run -n yolov12 python method02_advantage_regress_tinyConv/eval_image_protocol_bdd.py \
        --weight finetuned_bdd100k/<alpha0.2 ...>.pt --limit 2000
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402
import eval_baseline_kitti as B  # noqa: E402
from method02_advantage_regress_tinyConv.eval_video_bdd import BDD_MOT_EVAL_CLS  # noqa: E402

ID2NAME = {0: "pedestrian", 1: "rider", 2: "car", 3: "bus", 4: "truck",
           5: "bicycle", 6: "motorcycle", 7: "traffic light", 8: "traffic sign", 9: "train"}


def load_gt(txt, W, H):
    gts = []
    if not txt.exists():
        return gts
    for ln in txt.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        if c not in BDD_MOT_EVAL_CLS:
            continue
        cx, cy, w, h = (float(x) for x in p[1:5])
        gts.append((c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--img_root", default="/media/data/bdd100k_yolo/val")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=2000, help="0 = all 10k")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    B.EVAL_CLS = BDD_MOT_EVAL_CLS
    yolo = YOLO(args.weight, task="detect")
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super = [False] * N

    img_dir = Path(args.img_root) / "images"
    lbl_dir = Path(args.img_root) / "labels"
    imgs = sorted(img_dir.glob("*.jpg"))
    if args.limit > 0:
        imgs = imgs[:args.limit]

    matches, gt_count = [], defaultdict(int)
    for i, fp in enumerate(imgs):
        bgr = cv2.imread(str(fp))
        H, W = bgr.shape[:2]
        gts = load_gt(lbl_dir / f"{fp.stem}.txt", W, H)
        r = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                         iou=0.7, skip=skip_super, verbose=False, device=args.device)[0]
        preds = B.boxes_to_preds(r)
        m, gtc = B.match_frame_multi_iou(preds, gts)
        matches.extend(m)
        for c, n in gtc.items():
            gt_count[c] += n
        if (i + 1) % 500 == 0:
            print(f"[*] {i + 1}/{len(imgs)} images")

    ap50_per_cls, map50, map5095 = B.dataset_map_multi_iou(matches, gt_count)
    print(f"\n[*] det_20 val ({len(imgs)} imgs)  SUPER  conf={args.conf}  imgsz={tuple(args.imgsz)}")
    print(f"{'class':<14}{'n_gt':>9}{'AP50':>9}")
    for c in BDD_MOT_EVAL_CLS:
        print(f"{ID2NAME[c]:<14}{gt_count.get(c, 0):>9}{ap50_per_cls.get(c, 0.0):>9.4f}")
    print(f"{'MACRO mAP50':<14}{'':>9}{map50:>9.4f}")
    print(f"{'MACRO mAP':<14}{'':>9}{map5095:>9.4f}")


if __name__ == "__main__":
    main()
