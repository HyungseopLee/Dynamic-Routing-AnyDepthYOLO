"""
Per-image detection loss + post-NMS confidence stats on BDD100K (AnyDepth model).

Extends per_image_loss.py: for both Super-net and Base-net, additionally runs
NMS on the raw predictions and computes GT-free confidence features:
    n_pred_10,  mean_conf_10       (boxes with conf >= 0.10)
    n_pred_25,  mean_conf_25       (boxes with conf >= 0.25)
    n_pred_50,  mean_conf_50       (boxes with conf >= 0.50)
    n_pred_75,  mean_conf_75       (boxes with conf >= 0.75)
    top10_mean_conf                (mean of top-10 confs, NaN if <10 boxes)
    n_pred_all, mean_conf_all      (all post-NMS boxes, conf >= 0.001)

NMS is run with a low base threshold (0.001) so the 3 thresholds and top-10
are derived from the same post-NMS conf array with simple masks.

Usage:
    mkdir -p ./analysis/bdd100k-AnyDepth
    PYTHONPATH=$PWD \
    python tools/per_image_loss_conf.py \
        --weight ./finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.6_orig_mAP34.3_33.1.pt \
        --data bdd100k.yaml \
        --imgsz 1280 \
        --out ./analysis/bdd100k-AnyDepth/per_image_loss_conf.csv \
        2>&1 | tee ./analysis/bdd100k-AnyDepth/per_image_loss_conf.log
"""
import argparse
import csv
import math
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.ops import non_max_suppression

# thresholds + top-k
THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
TOPK = 10
NMS_BASE_CONF = 0.001   # keep almost everything so we can mask later
NMS_IOU = 0.7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--data", default="bdd100k.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="analysis/bdd100k-AnyDepth/per_image_loss_conf.csv")
    p.add_argument("--limit", type=int, default=0, help="debug: cap #images")
    return p.parse_args()


def loss_items_to_tuple(loss_items):
    box, cls_l, dfl = [float(x) for x in loss_items.detach().cpu().tolist()]
    return box, cls_l, dfl, box + cls_l + dfl


def conf_stats(conf_tensor):
    """Return per-threshold (n, mean) pairs, then top10_mean, then n_all, mean_all."""
    out = []
    for t in THRESHOLDS:
        mask = conf_tensor >= t
        n = int(mask.sum().item())
        m = float(conf_tensor[mask].mean().item()) if n > 0 else float("nan")
        out += [n, m]
    if conf_tensor.numel() >= TOPK:
        topk = torch.topk(conf_tensor, k=TOPK).values
        out.append(float(topk.mean().item()))
    else:
        out.append(float("nan"))
    # all post-NMS boxes (no threshold above NMS_BASE_CONF)
    n_all = int(conf_tensor.numel())
    m_all = float(conf_tensor.mean().item()) if n_all > 0 else float("nan")
    out += [n_all, m_all]
    return out


def get_raw_pred(out):
    """model.predict() can return Tensor or (Tensor, aux). Return the raw NMS-input tensor."""
    return out[0] if isinstance(out, (list, tuple)) else out


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    yolo = YOLO(args.weight, task="detect")
    model = yolo.model.to(device).eval()

    num_skip = getattr(model, "num_skippable_layers", 0)
    is_anydepth = num_skip > 0
    print(f"[*] AnyDepth: {is_anydepth} (num_skippable_layers={num_skip})")
    if not is_anydepth:
        raise SystemExit("This script expects an AnyDepth model (num_skippable_layers > 0).")

    data = check_det_dataset(args.data)
    img_path = data[args.split]

    cfg = get_cfg(DEFAULT_CFG_DICT, overrides={"imgsz": args.imgsz, "task": "detect", "mode": "val"})
    dataset = build_yolo_dataset(cfg, img_path, batch=1, data=data, mode="val",
                                 stride=int(max(model.stride)), rect=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    )

    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils import IterableSimpleNamespace
    merged_args = dict(DEFAULT_CFG_DICT)
    ckpt_args = model.args if isinstance(model.args, dict) else vars(model.args)
    merged_args.update(ckpt_args)
    model.args = IterableSimpleNamespace(**merged_args)
    criterion = v8DetectionLoss(model)
    nc = int(getattr(model, "nc", 0)) or len(getattr(model, "names", []))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip_super = [False] * num_skip
    skip_base  = [True]  * num_skip

    conf_cols = []
    for t in THRESHOLDS:
        tag = f"{int(round(t*100)):02d}"
        conf_cols += [f"n_pred_{tag}", f"mean_conf_{tag}"]
    conf_cols += ["top10_mean_conf", "n_pred_all", "mean_conf_all"]

    header = ["stem",
              "loss_super", "box_super", "cls_super", "dfl_super",
              "loss_base",  "box_base",  "cls_base",  "dfl_base",
              "loss_diff", "n_gt"] \
             + [c + "_super" for c in conf_cols] \
             + [c + "_base"  for c in conf_cols]

    rows = []
    n = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    with torch.no_grad():
        pbar = tqdm(loader, total=n)
        for i, batch in enumerate(pbar):
            if args.limit and i >= args.limit:
                break
            batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255.0
            for k in ("cls", "bboxes", "batch_idx"):
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)

            # --- Super-net ---
            preds_super = model.predict(batch["img"], skip=skip_super)
            _, items_super = criterion(preds_super, batch)
            box_s, cls_s, dfl_s, total_s = loss_items_to_tuple(items_super)
            raw_s = get_raw_pred(preds_super)
            det_s = non_max_suppression(raw_s, conf_thres=NMS_BASE_CONF,
                                        iou_thres=NMS_IOU, nc=nc)[0]
            conf_s_tensor = det_s[:, 4] if det_s is not None and det_s.numel() else torch.empty(0, device=device)
            stats_s = conf_stats(conf_s_tensor)

            # --- Base-net ---
            preds_base = model.predict(batch["img"], skip=skip_base)
            _, items_base = criterion(preds_base, batch)
            box_b, cls_b, dfl_b, total_b = loss_items_to_tuple(items_base)
            raw_b = get_raw_pred(preds_base)
            det_b = non_max_suppression(raw_b, conf_thres=NMS_BASE_CONF,
                                        iou_thres=NMS_IOU, nc=nc)[0]
            conf_b_tensor = det_b[:, 4] if det_b is not None and det_b.numel() else torch.empty(0, device=device)
            stats_b = conf_stats(conf_b_tensor)

            stem = Path(batch["im_file"][0]).stem
            n_gt = int(batch["cls"].numel())
            rows.append([stem,
                         total_s, box_s, cls_s, dfl_s,
                         total_b, box_b, cls_b, dfl_b,
                         total_b - total_s, n_gt] + stats_s + stats_b)
            if i % 100 == 0:
                pbar.set_postfix(s=f"{total_s:.2f}", b=f"{total_b:.2f}",
                                 nS=stats_s[0], nB=stats_b[0])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[*] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
