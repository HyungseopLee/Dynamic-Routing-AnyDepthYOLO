"""
Per-frame confidence stats + per-image mAP@[0.5:0.95] on KITTI Tracking
for Super-net and Base-net (AnyDepth, BDD-finetuned weight).

Goal: replicate the conf-AP correlation analysis we did on BDD100K, this time
on KITTI Tracking sequences (all 21 train sequences flattened).

For each frame and for each of {Super (skip=FFFF), Base (skip=TTTT)}:
  - post-NMS conf stats: n_pred_{10,25,50,75,all}, mean_conf_{10,25,50,75,all},
                         top10_mean_conf
  - per-image mAP@[0.5:0.95]  (threshold-free, IoU grid 0.5..0.95)

Only BDD classes that have KITTI counterparts are evaluated:
    person, rider, car, truck, train  (5 classes)

Usage:
    mkdir -p ./analysis/kitti-AnyDepth
    PYTHONPATH=$PWD python tools/per_image_kitti.py \
        --weight ./finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.6_orig_mAP34.3_33.1.pt \
        --kitti_root /media/data/kitti-tracking \
        --imgsz 1242 375 \
        --out ./analysis/kitti-AnyDepth/per_image_kitti.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import logging
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.metrics import box_iou, compute_ap

LOGGER.setLevel(logging.ERROR)   # suppress imgsz multiple-of-stride warnings


# ---- reuse BDD constants and helpers ----
THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
TOPK = 10
NMS_BASE_CONF = 0.001
NMS_IOU = 0.7
IOU_GRID = tuple(round(x, 2) for x in np.arange(0.5, 1.0, 0.05))

# KITTI -> BDD class mapping (same as eval_video_kitti.py)
KITTI_TO_BDD = {
    "Car": 2, "Van": 2, "Truck": 4,
    "Pedestrian": 0, "Person_sitting": 0,
    "Cyclist": 1, "Tram": 9,
}
EVAL_CLS = sorted(set(KITTI_TO_BDD.values()))   # [0, 1, 2, 4, 9]


def conf_stats(conf_tensor):
    out = []
    for t in THRESHOLDS:
        mask = conf_tensor >= t
        n = int(mask.sum().item())
        m = float(conf_tensor[mask].mean().item()) if n > 0 else float("nan")
        out += [n, m]
    if conf_tensor.numel() >= TOPK:
        out.append(float(torch.topk(conf_tensor, k=TOPK).values.mean().item()))
    else:
        out.append(float("nan"))
    n_all = int(conf_tensor.numel())
    m_all = float(conf_tensor.mean().item()) if n_all > 0 else float("nan")
    out += [n_all, m_all]
    return out


def per_image_ap5095(det, gt_xyxy, gt_cls, iou_grid=IOU_GRID, eval_cls=EVAL_CLS):
    """COCO-style mean AP averaged over iou_grid, restricted to eval_cls."""
    # filter GT to eval classes
    if gt_cls.numel() > 0:
        keep_gt = torch.zeros_like(gt_cls, dtype=torch.bool)
        for c in eval_cls:
            keep_gt |= (gt_cls == c)
        gt_cls = gt_cls[keep_gt]
        gt_xyxy = gt_xyxy[keep_gt]
    n_gt = int(gt_xyxy.shape[0])
    if n_gt == 0:
        return float("nan")
    if det is None or det.numel() == 0:
        return 0.0
    # filter det to eval classes too (FP outside eval set wouldn't count)
    pred_cls_all = det[:, 5].long()
    keep_pred = torch.zeros_like(pred_cls_all, dtype=torch.bool)
    for c in eval_cls:
        keep_pred |= (pred_cls_all == c)
    det = det[keep_pred]
    if det.numel() == 0:
        return 0.0

    order = torch.argsort(det[:, 4], descending=True)
    det = det[order]
    pred_cls = det[:, 5].long()
    iou = box_iou(det[:, :4], gt_xyxy)
    cls_match = pred_cls.unsqueeze(1) == gt_cls.unsqueeze(0)
    iou_masked = (iou * cls_match.float()).cpu().numpy()
    pred_cls_np = pred_cls.cpu().numpy()
    gt_cls_np = gt_cls.cpu().numpy()

    classes = np.unique(gt_cls_np)
    ap_per_iou = []
    for iou_th in iou_grid:
        aps = []
        for c in classes:
            gt_idx = np.where(gt_cls_np == c)[0]
            pred_idx = np.where(pred_cls_np == c)[0]
            n_gt_c = len(gt_idx)
            if len(pred_idx) == 0:
                aps.append(0.0); continue
            iou_c = iou_masked[pred_idx][:, gt_idx]
            n_pred_c = iou_c.shape[0]
            matched = np.zeros(n_gt_c, dtype=bool)
            tp = np.zeros(n_pred_c, dtype=np.float32)
            for j in range(n_pred_c):
                ious_j = iou_c[j].copy(); ious_j[matched] = 0
                best = int(np.argmax(ious_j))
                if ious_j[best] >= iou_th:
                    matched[best] = True; tp[j] = 1.0
            fp = 1.0 - tp
            tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
            recall_c = tp_cum / max(n_gt_c, 1)
            precision_c = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap_c, _, _ = compute_ap(recall_c, precision_c)
            aps.append(float(ap_c))
        ap_per_iou.append(float(np.mean(aps)) if aps else 0.0)
    return float(np.mean(ap_per_iou))


def parse_kitti_labels(label_path):
    """{frame_idx: [(cls_bdd, x1, y1, x2, y2), ...]} in original-image pixels."""
    gt = defaultdict(list)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10: continue
            frame_idx = int(parts[0])
            obj_type = parts[2]
            if obj_type not in KITTI_TO_BDD: continue
            cls = KITTI_TO_BDD[obj_type]
            x1, y1, x2, y2 = map(float, parts[6:10])
            gt[frame_idx].append((cls, x1, y1, x2, y2))
    return gt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--kitti_root", default="/media/data/kitti-tracking")
    p.add_argument("--imgsz", type=int, nargs=2, default=[1242, 375])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="analysis/kitti-AnyDepth/per_image_kitti.csv")
    p.add_argument("--sequences", type=str, nargs="*", default=None,
                   help="subset of sequence ids (e.g. 0000 0001). default: all")
    p.add_argument("--limit", type=int, default=0, help="cap frames per seq (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    yolo = YOLO(args.weight, task="detect")
    model = yolo.model.to(device).eval()
    num_skip = getattr(model, "num_skippable_layers", 0)
    print(f"[*] AnyDepth: {num_skip > 0} (num_skippable_layers={num_skip})")
    if num_skip <= 0:
        raise SystemExit("AnyDepth model required.")
    nc = int(getattr(model, "nc", 0)) or len(getattr(model, "names", []))
    skip_super = [False] * num_skip
    skip_base  = [True]  * num_skip

    # sequences
    label_dir = Path(args.kitti_root) / "training" / "label_02"
    img_root = Path(args.kitti_root) / "training" / "image_02"
    if args.sequences:
        seqs = args.sequences
    else:
        seqs = sorted(f.stem for f in label_dir.glob("*.txt"))
    print(f"[*] sequences: {len(seqs)}")

    # CSV header
    conf_cols = []
    for t in THRESHOLDS:
        tag = f"{int(round(t*100)):02d}"
        conf_cols += [f"n_pred_{tag}", f"mean_conf_{tag}"]
    conf_cols += ["top10_mean_conf", "n_pred_all", "mean_conf_all"]
    header = ["stem", "seq", "frame_idx", "n_gt"] \
             + [c + "_super" for c in conf_cols] \
             + [c + "_base"  for c in conf_cols] \
             + ["ap5095_super", "ap5095_base"]

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    with torch.no_grad():
        for seq in seqs:
            label_path = label_dir / f"{seq}.txt"
            seq_img_dir = img_root / seq
            if not label_path.exists() or not seq_img_dir.exists():
                print(f"  [!] skip {seq}: missing image dir or labels")
                continue
            gt_by_frame = parse_kitti_labels(label_path)
            frame_files = sorted(seq_img_dir.glob("*.png")) or sorted(seq_img_dir.glob("*.jpg"))
            n = len(frame_files) if args.limit <= 0 else min(args.limit, len(frame_files))
            pbar = tqdm(frame_files[:n], desc=f"seq {seq}", total=n)
            for fpath in pbar:
                frame_idx = int(fpath.stem)
                bgr = cv2.imread(str(fpath))
                if bgr is None: continue
                # GT in original pixels
                gt_list = gt_by_frame.get(frame_idx, [])
                gt_arr = np.array([[x1, y1, x2, y2] for (_, x1, y1, x2, y2) in gt_list],
                                  dtype=np.float32) if gt_list else np.zeros((0, 4), dtype=np.float32)
                gt_cls_arr = np.array([c for (c, *_) in gt_list], dtype=np.int64) \
                             if gt_list else np.zeros((0,), dtype=np.int64)

                # ultralytics handles resize/letterbox internally on imgsz=(h,w)
                results_super = yolo.predict(source=bgr, imgsz=tuple(args.imgsz),
                                             conf=NMS_BASE_CONF, iou=NMS_IOU,
                                             skip=skip_super, verbose=False, device=device)
                results_base  = yolo.predict(source=bgr, imgsz=tuple(args.imgsz),
                                             conf=NMS_BASE_CONF, iou=NMS_IOU,
                                             skip=skip_base,  verbose=False, device=device)
                # box outputs are in original-image pixel coords (ultralytics rescales for us)
                r_s, r_b = results_super[0], results_base[0]

                def boxes_to_det(r):
                    if r.boxes is None or len(r.boxes) == 0:
                        return torch.empty(0, 6, device=device)
                    xyxy = r.boxes.xyxy.to(device)
                    conf = r.boxes.conf.to(device).unsqueeze(1)
                    cls  = r.boxes.cls.to(device).unsqueeze(1)
                    return torch.cat([xyxy, conf, cls], dim=1)

                det_s = boxes_to_det(r_s)
                det_b = boxes_to_det(r_b)
                conf_s_t = det_s[:, 4] if det_s.numel() else torch.empty(0, device=device)
                conf_b_t = det_b[:, 4] if det_b.numel() else torch.empty(0, device=device)
                stats_s = conf_stats(conf_s_t)
                stats_b = conf_stats(conf_b_t)

                gt_xyxy_t = torch.from_numpy(gt_arr).to(device)
                gt_cls_t  = torch.from_numpy(gt_cls_arr).to(device)
                ap_s = per_image_ap5095(det_s, gt_xyxy_t, gt_cls_t)
                ap_b = per_image_ap5095(det_b, gt_xyxy_t, gt_cls_t)

                stem = f"{seq}_{frame_idx:06d}"
                rows.append([stem, seq, frame_idx, int(gt_cls_t.numel())]
                            + stats_s + stats_b + [ap_s, ap_b])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"\n[*] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
