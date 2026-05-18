"""
Per-image loss + post-NMS confidence + precision/recall + mAP@[0.5:0.95]
for Super-net and Base-net (AnyDepth) on BDD100K.

For each image and for each of {Super-net (no skip), Base-net (skip all)}:
  - loss components from v8DetectionLoss: box / cls / dfl / total
  - post-NMS confidence stats (NMS at conf>=0.001, IoU=0.7):
        n_pred_{10,25,50,75,all}, mean_conf_{10,25,50,75,all},
        top10_mean_conf
  - PR at conf>=0.25, IoU>=0.5: tp / fp / fn / precision / recall
  - mAP@[0.5:0.95] threshold-free (10 IoUs averaged)

Outputs a single wide CSV; all downstream analyze_* scripts read from it.

Usage:
    PYTHONPATH=$PWD python tools/per_image_loss_pr_conf.py \
        --weight ./finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.6_orig_mAP34.3_33.1.pt \
        --data bdd100k.yaml --imgsz 1280 \
        --out ./analysis/bdd100k-AnyDepth/per_image_loss_pr_conf.csv
"""
import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
from ultralytics.utils.metrics import box_iou, compute_ap
from ultralytics.utils.loss import v8DetectionLoss
import numpy as np


THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
TOPK = 10
NMS_BASE_CONF = 0.001
NMS_IOU = 0.7
PR_CONF = 0.25       # τ for precision/recall
PR_IOU = 0.5         # IoU threshold for TP (P/R only)
IOU_GRID = tuple(round(x, 2) for x in np.arange(0.5, 1.0, 0.05))  # 0.5, 0.55, ..., 0.95


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--data", default="bdd100k.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="analysis/bdd100k-AnyDepth/per_image_loss_pr_conf.csv")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def loss_items_to_tuple(li):
    box, cls_l, dfl = [float(x) for x in li.detach().cpu().tolist()]
    return box, cls_l, dfl, box + cls_l + dfl


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


def per_image_ap5095(det, gt_xyxy, gt_cls, iou_grid=IOU_GRID):
    """Per-image COCO-style mAP@[0.5:0.95]: average of per-IoU mean(per-class AP).

    No conf threshold — uses all post-NMS boxes (threshold-free).
    For each IoU in iou_grid: compute mean AP across classes present in GT.
    Return mean over iou_grid.  NaN if image has no GT.
    """
    n_gt = int(gt_xyxy.shape[0])
    if n_gt == 0:
        return float("nan")
    if det is None or det.numel() == 0:
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
            iou_c = iou_masked[pred_idx][:, gt_idx]   # (n_pred_c, n_gt_c)
            n_pred_c = iou_c.shape[0]
            matched = np.zeros(n_gt_c, dtype=bool)
            tp = np.zeros(n_pred_c, dtype=np.float32)
            for j in range(n_pred_c):
                ious_j = iou_c[j].copy()
                ious_j[matched] = 0
                best = int(np.argmax(ious_j))
                if ious_j[best] >= iou_th:
                    matched[best] = True
                    tp[j] = 1.0
            fp = 1.0 - tp
            tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
            recall_c = tp_cum / max(n_gt_c, 1)
            precision_c = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap_c, _, _ = compute_ap(recall_c, precision_c)
            aps.append(float(ap_c))
        ap_per_iou.append(float(np.mean(aps)) if aps else 0.0)
    return float(np.mean(ap_per_iou))


def pr_stats(det, gt_xyxy, gt_cls, conf_th=PR_CONF, iou_th=PR_IOU):
    """
    det:     (N, 6) tensor [x1, y1, x2, y2, conf, cls] in image-pixel coords
    gt_xyxy: (M, 4) tensor
    gt_cls:  (M,)   tensor

    Returns (tp, fp, fn, precision, recall).
    """
    n_gt = int(gt_xyxy.shape[0])
    if det is None or det.numel() == 0:
        return 0, 0, n_gt, float("nan") if n_gt == 0 else 0.0, 0.0 if n_gt > 0 else float("nan")

    # filter by conf threshold
    keep = det[:, 4] >= conf_th
    det = det[keep]
    n_pred = int(det.shape[0])
    if n_pred == 0:
        return 0, 0, n_gt, float("nan") if n_gt == 0 else 0.0, 0.0 if n_gt > 0 else float("nan")
    if n_gt == 0:
        return 0, n_pred, 0, 0.0, float("nan")

    # sort detections by conf desc
    order = torch.argsort(det[:, 4], descending=True)
    det = det[order]
    pred_cls = det[:, 5].long()
    iou = box_iou(det[:, :4], gt_xyxy)        # (n_pred, n_gt)
    # class match mask
    cls_match = pred_cls.unsqueeze(1) == gt_cls.unsqueeze(0)
    iou = iou * cls_match.float()

    matched_gt = torch.zeros(n_gt, dtype=torch.bool, device=det.device)
    tp = 0
    for i in range(n_pred):
        ious_i = iou[i]
        ious_i = ious_i.masked_fill(matched_gt, 0)
        best = int(torch.argmax(ious_i))
        if float(ious_i[best]) >= iou_th:
            matched_gt[best] = True
            tp += 1
    fp = n_pred - tp
    fn = n_gt - int(matched_gt.sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return tp, fp, fn, float(precision), float(recall)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    yolo = YOLO(args.weight, task="detect")
    model = yolo.model.to(device).eval()
    num_skip = getattr(model, "num_skippable_layers", 0)
    if num_skip <= 0:
        raise SystemExit("AnyDepth model required.")

    data = check_det_dataset(args.data)
    img_path = data[args.split]
    cfg = get_cfg(DEFAULT_CFG_DICT, overrides={"imgsz": args.imgsz, "task": "detect", "mode": "val"})
    dataset = build_yolo_dataset(cfg, img_path, batch=1, data=data, mode="val",
                                 stride=int(max(model.stride)), rect=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    merged = dict(DEFAULT_CFG_DICT)
    ckpt_args = model.args if isinstance(model.args, dict) else vars(model.args)
    merged.update(ckpt_args)
    model.args = IterableSimpleNamespace(**merged)
    criterion = v8DetectionLoss(model)
    nc = int(getattr(model, "nc", 0)) or len(getattr(model, "names", []))

    skip_super = [False] * num_skip
    skip_base  = [True]  * num_skip

    conf_cols = []
    for t in THRESHOLDS:
        tag = f"{int(round(t*100)):02d}"
        conf_cols += [f"n_pred_{tag}", f"mean_conf_{tag}"]
    conf_cols += ["top10_mean_conf", "n_pred_all", "mean_conf_all"]
    pr_cols = ["tp", "fp", "fn", "precision", "recall", "ap5095"]

    header = ["stem",
              "loss_super", "box_super", "cls_super", "dfl_super",
              "loss_base",  "box_base",  "cls_base",  "dfl_base",
              "loss_diff", "n_gt"] \
             + [c + "_super" for c in conf_cols] \
             + [c + "_base"  for c in conf_cols] \
             + [c + "_super" for c in pr_cols] \
             + [c + "_base"  for c in pr_cols]

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    with torch.no_grad():
        pbar = tqdm(loader, total=n)
        for i, batch in enumerate(pbar):
            if args.limit and i >= args.limit: break
            batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255.0
            for k in ("cls", "bboxes", "batch_idx"):
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)
            img_h, img_w = batch["img"].shape[-2], batch["img"].shape[-1]

            # GT for PR (xywh normalized -> xyxy in pixel coords of resized img)
            gt_xywh = batch["bboxes"].clone()
            gt_xywh[:, 0] *= img_w; gt_xywh[:, 2] *= img_w
            gt_xywh[:, 1] *= img_h; gt_xywh[:, 3] *= img_h
            gt_xyxy = xywh2xyxy(gt_xywh)
            gt_cls = batch["cls"].squeeze(-1).long() if batch["cls"].numel() else batch["cls"].long()

            # --- Super ---
            preds_super = model.predict(batch["img"], skip=skip_super)
            _, items_super = criterion(preds_super, batch)
            box_s, cls_s, dfl_s, total_s = loss_items_to_tuple(items_super)
            raw_s = preds_super[0] if isinstance(preds_super, (list, tuple)) else preds_super
            det_s = non_max_suppression(raw_s, conf_thres=NMS_BASE_CONF,
                                        iou_thres=NMS_IOU, nc=nc)[0]
            conf_s = det_s[:, 4] if det_s is not None and det_s.numel() else torch.empty(0, device=device)
            stats_s = conf_stats(conf_s)
            tp_s, fp_s, fn_s, p_s, r_s = pr_stats(det_s, gt_xyxy, gt_cls)
            ap_s = per_image_ap5095(det_s, gt_xyxy, gt_cls)

            # --- Base ---
            preds_base = model.predict(batch["img"], skip=skip_base)
            _, items_base = criterion(preds_base, batch)
            box_b, cls_b, dfl_b, total_b = loss_items_to_tuple(items_base)
            raw_b = preds_base[0] if isinstance(preds_base, (list, tuple)) else preds_base
            det_b = non_max_suppression(raw_b, conf_thres=NMS_BASE_CONF,
                                        iou_thres=NMS_IOU, nc=nc)[0]
            conf_b = det_b[:, 4] if det_b is not None and det_b.numel() else torch.empty(0, device=device)
            stats_b = conf_stats(conf_b)
            tp_b, fp_b, fn_b, p_b, r_b = pr_stats(det_b, gt_xyxy, gt_cls)
            ap_b = per_image_ap5095(det_b, gt_xyxy, gt_cls)

            stem = Path(batch["im_file"][0]).stem
            n_gt = int(batch["cls"].numel())
            rows.append([stem,
                         total_s, box_s, cls_s, dfl_s,
                         total_b, box_b, cls_b, dfl_b,
                         total_b - total_s, n_gt]
                        + stats_s + stats_b
                        + [tp_s, fp_s, fn_s, p_s, r_s, ap_s]
                        + [tp_b, fp_b, fn_b, p_b, r_b, ap_b])
            if i % 100 == 0:
                pbar.set_postfix(s=f"{total_s:.2f}", b=f"{total_b:.2f}",
                                 pS=f"{p_s:.2f}", rS=f"{r_s:.2f}")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[*] wrote {len(rows)} rows -> {out_path}")
    print(f"[*] PR computed at conf>={PR_CONF}, IoU>={PR_IOU}")


if __name__ == "__main__":
    main()
