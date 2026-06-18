"""Condition-stratified BASE vs SUPER AP on BDD100K det val (10k images).

Independent reproduction, in OUR AnyDepth-YOLO setting, of the condition-stratified
analysis that motivates per-image depth routing: the SUPER-BASE AP gap is NOT
uniform across scenes, so a single global operating point is wasteful. For every
val image we run BOTH paths once, then aggregate AP@[.50:.95] within each
scene-attribute group (time of day, weather, scene type, crowd density). The
per-group SUPER-BASE gap quantifies where the deep path actually pays off.

AP is the official ultralytics COCO metric (same evaluator that produced the
checkpoint's headline mAP): per-image TP matrices over 10 IoU thresholds are
accumulated per group and scored with ultralytics.utils.metrics.ap_per_class.

Attributes come from the official bdd100k_labels_images_val.json (string labels);
crowd density is split at the per-image object-count median (low = sparse).

    conda run -n yolov12 python -m method02_advantage_regress_tinyConv.eval_condition_stratified_bdd \
        --weight finetuned_bdd100k/<anydepth>.pt --limit 0
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa
from ultralytics.utils.metrics import ap_per_class, box_iou  # noqa
from method02_advantage_regress_tinyConv.eval_video_bdd import BDD_MOT_EVAL_CLS  # noqa

OUT = Path(__file__).resolve().parent / "outputs"
IOUV = torch.linspace(0.5, 0.95, 10)  # COCO IoU sweep

GROUPS = [
    ("Time of day", "timeofday", ["daytime", "night", "dawn/dusk"]),
    ("Weather",     "weather",   ["clear", "overcast", "rainy", "snowy"]),
    ("Scene type",  "scene",     ["highway", "city street", "residential"]),
]


def load_gt_cls(txt, W, H, keep):
    """YOLO-txt GT -> (boxes[M,4] xyxy, cls[M]); keep only class ids in `keep`."""
    boxes, cls = [], []
    if txt.exists():
        for ln in txt.read_text().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c not in keep:
                continue
            cx, cy, w, h = (float(x) for x in p[1:5])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H])
            cls.append(c)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(cls, dtype=np.float32))


def match(det, gt_boxes, gt_cls):
    """correct[N,10] TP matrix (ultralytics _process_batch logic)."""
    n = det.shape[0]
    correct = np.zeros((n, 10), dtype=bool)
    if n == 0 or gt_boxes.shape[0] == 0:
        return correct
    iou = box_iou(torch.from_numpy(gt_boxes), det[:, :4])  # (M,N)
    correct_class = torch.from_numpy(gt_cls)[:, None] == det[:, 5].cpu()
    iou = (iou * correct_class).cpu().numpy()
    for i, thr in enumerate(IOUV.tolist()):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def group_ap(idxs, per_img):
    """Official mAP@.50 / mAP@[.50:.95] over a subset, per path."""
    tp = np.concatenate([per_img[i]["tp"] for i in idxs]) if idxs else np.zeros((0, 10), bool)
    conf = np.concatenate([per_img[i]["conf"] for i in idxs]) if idxs else np.zeros(0)
    pcls = np.concatenate([per_img[i]["pcls"] for i in idxs]) if idxs else np.zeros(0)
    tcls = np.concatenate([per_img[i]["tcls"] for i in idxs]) if idxs else np.zeros(0)
    if tcls.shape[0] == 0:
        return 0.0, 0.0
    ap = ap_per_class(tp, conf, pcls, tcls, names={i: str(i) for i in range(10)})[5]
    return float(ap[:, 0].mean()), float(ap.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--img_root", default="/media/data/bdd100k_yolo/val")
    ap.add_argument("--labels_json",
                    default="/media/data/bdd100k_yolo/bdd100k_labels_images_val.json")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 10k images")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--classes", choices=["movable", "all"], default="all",
                    help="all = 10 classes (matches headline mAP); movable = 8 MOT classes")
    ap.add_argument("--out", default=str(OUT / "bdd100k/eval/condition_stratified.json"))
    args = ap.parse_args()

    keep = set(range(10)) if args.classes == "all" else set(BDD_MOT_EVAL_CLS)
    dev = args.device if torch.cuda.is_available() else "cpu"
    img_root = Path(args.img_root)
    lbl_dir = img_root / "labels"

    meta = json.loads(Path(args.labels_json).read_text())
    if args.limit:
        meta = meta[:args.limit]
    attrs = [m.get("attributes", {}) for m in meta]
    names = [m["name"] for m in meta]
    n = len(names)

    yolo = YOLO(args.weight, task="detect")
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip = {"base": [True] * N, "super": [False] * N}

    rec = {"super": [None] * n, "base": [None] * n}
    obj_count = np.zeros(n, dtype=int)

    for i in range(n):
        stem = Path(names[i]).stem
        bgr = cv2.imread(str(img_root / "images" / f"{stem}.jpg"))
        if bgr is None:
            raise FileNotFoundError(img_root / "images" / f"{stem}.jpg")
        H, W = bgr.shape[:2]
        gt_boxes, gt_cls = load_gt_cls(lbl_dir / f"{stem}.txt", W, H, keep)
        obj_count[i] = gt_cls.shape[0]
        for tag in ("super", "base"):
            r = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                             iou=0.7, skip=skip[tag], verbose=False, device=dev)[0]
            if r.boxes is not None and len(r.boxes):
                det = torch.cat([r.boxes.xyxy, r.boxes.conf[:, None],
                                 r.boxes.cls[:, None]], 1).cpu()
                det = det[np.isin(det[:, 5].numpy().astype(int), list(keep))]
            else:
                det = torch.zeros((0, 6))
            rec[tag][i] = {"tp": match(det, gt_boxes, gt_cls),
                           "conf": det[:, 4].numpy(), "pcls": det[:, 5].numpy(),
                           "tcls": gt_cls}
        if (i + 1) % 250 == 0:
            print(f"[*] {i + 1}/{n}", flush=True)

    med = int(np.median(obj_count))
    rows = []

    def add_row(heading, label, idxs):
        if len(idxs) == 0:
            return
        s50, s = group_ap(idxs, rec["super"])
        b50, b = group_ap(idxs, rec["base"])
        rows.append({"heading": heading, "label": label, "n": len(idxs),
                     "super": s, "base": b, "gap": s - b,
                     "super50": s50, "base50": b50})
        print(f"  {label:14s} N={len(idxs):5d}  super={s:.4f} base={b:.4f} "
              f"gap={s - b:+.4f}", flush=True)

    add_row("All", "All", list(range(n)))
    for heading, key, vals in GROUPS:
        for v in vals:
            add_row(heading, v, [i for i in range(n) if attrs[i].get(key) == v])
    add_row("Crowd density", f"Low (<= {med})",
            [i for i in range(n) if obj_count[i] <= med])
    add_row("Crowd density", f"High (> {med})",
            [i for i in range(n) if obj_count[i] > med])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"rows": rows, "n": n, "crowd_median": med,
         "meta": {"weight": args.weight, "conf": args.conf, "imgsz": args.imgsz,
                  "classes": args.classes, "metric": "ultralytics_coco"}}, indent=1))
    print(f"\n[*] {n} images ({args.classes} classes), crowd median={med} -> {args.out}")


if __name__ == "__main__":
    main()
