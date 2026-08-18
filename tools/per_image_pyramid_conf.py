"""
Per-image per-pyramid-level (P3 / P4 / P5) confidence stats for
AnyDepth-YOLO Super-net and Base-net, plus GT-side area-bin stats.

For each image and each {Super (no skip), Base (skip all)}:
  - From the Detect head's RAW per-level tensors (no NMS), we extract
    the per-anchor max-class confidence (sigmoid of cls logits) for
    each of P3 / P4 / P5, and report:
        n_anchors_{P3,P4,P5}     (deterministic; depends on imgsz)
        n_pred_ge25_{P3,P4,P5}   (# anchors with max-conf >= 0.25)
        mean_conf_all_{P3,P4,P5}
        mean_conf_ge10_{P3,P4,P5}
        top{10,20,30}_mean_conf_{P3,P4,P5}
        max_conf_{P3,P4,P5}

GT side (no NMS dependency):
  - COCO-style area bins on normalized (xywh) GT boxes converted to px:
        n_gt_small    (area  < 32^2)
        n_gt_medium   (32^2 <= area < 96^2)
        n_gt_large    (area >= 96^2)
        n_gt_total
        gt_med_area   (median GT area, px^2)

Writes one wide CSV; downstream analysis script reads from it.

Usage:
    PYTHONPATH=$PWD python tools/per_image_pyramid_conf.py \
        --weight ./results/step1_finetune/pretrained_coco/yolo-ad-v12-orig_small_0.479_0.451.pt \
        --data bdd100k.yaml --imgsz 1280 \
        --out ./analysis/bdd100k-AnyDepth/per_image_pyramid_conf.csv
"""
import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.data.utils import check_det_dataset


THRESHOLDS = (0.10, 0.25)
TOPKS = (10, 20, 30)
LEVELS = ("P3", "P4", "P5")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--data", default="bdd100k.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="analysis/bdd100k-AnyDepth/per_image_pyramid_conf.csv")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def per_level_conf_stats(level_tensor, reg_max, nc):
    """level_tensor: (1, nc + 4*reg_max, H, W). Return dict of conf stats for this level.
    Confidence = max over classes of sigmoid(cls_logit). No box decoding.
    """
    _, C, H, W = level_tensor.shape
    cls_logits = level_tensor[:, reg_max * 4:, :, :]              # (1, nc, H, W)
    cls_prob = cls_logits.sigmoid()                                # (1, nc, H, W)
    per_anchor_conf, _ = cls_prob.max(dim=1)                       # (1, H, W)
    flat = per_anchor_conf.view(-1)                                # (H*W,)

    out = {"n_anchors": int(flat.numel())}
    out["mean_conf_all"] = float(flat.mean().item())
    out["max_conf"]      = float(flat.max().item())
    # threshold-conditional means
    for t in THRESHOLDS:
        mask = flat >= t
        out[f"n_pred_ge{int(round(t*100)):02d}"] = int(mask.sum().item())
        out[f"mean_conf_ge{int(round(t*100)):02d}"] = (
            float(flat[mask].mean().item()) if int(mask.sum()) > 0 else float("nan")
        )
    # top-k means
    for k in TOPKS:
        k_eff = min(k, flat.numel())
        topv = torch.topk(flat, k_eff).values
        out[f"top{k}_mean_conf"] = float(topv.mean().item())
    return out


def gt_area_bins(bboxes_xywh_norm, img_h, img_w):
    """Return dict with GT area bin counts in COCO style (px^2).
       small < 32^2,  32^2 <= medium < 96^2,  large >= 96^2.
    """
    if bboxes_xywh_norm.numel() == 0:
        return {"n_gt_small": 0, "n_gt_medium": 0, "n_gt_large": 0,
                "n_gt_total": 0, "gt_med_area": float("nan")}
    w_px = bboxes_xywh_norm[:, 2] * img_w
    h_px = bboxes_xywh_norm[:, 3] * img_h
    area = w_px * h_px
    small  = int((area <  32 ** 2).sum().item())
    medium = int(((area >= 32 ** 2) & (area < 96 ** 2)).sum().item())
    large  = int((area >= 96 ** 2).sum().item())
    return {"n_gt_small": small, "n_gt_medium": medium, "n_gt_large": large,
            "n_gt_total": int(area.numel()),
            "gt_med_area": float(area.median().item())}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    yolo = YOLO(args.weight, task="detect")
    model = yolo.model.to(device).eval()
    num_skip = getattr(model, "num_skippable_layers", 0)
    if num_skip <= 0:
        raise SystemExit("AnyDepth model required.")

    # Detect head metadata for slicing cls/box logits
    det_head = model.model[-1]
    nc = int(getattr(det_head, "nc", getattr(model, "nc", 0)))
    reg_max = int(getattr(det_head, "reg_max", 16))
    nl = int(getattr(det_head, "nl", 3))
    if nl != len(LEVELS):
        raise SystemExit(f"Expected {len(LEVELS)} detection levels, got {nl}")

    data = check_det_dataset(args.data)
    img_path = data[args.split]
    cfg = get_cfg(DEFAULT_CFG_DICT, overrides={"imgsz": args.imgsz, "task": "detect", "mode": "val"})
    dataset = build_yolo_dataset(cfg, img_path, batch=1, data=data, mode="val",
                                 stride=int(max(model.stride)), rect=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    skip_super = [False] * num_skip
    skip_base  = [True]  * num_skip

    # column builder
    stat_names = ["n_anchors", "mean_conf_all", "max_conf"] \
        + [f"n_pred_ge{int(round(t*100)):02d}" for t in THRESHOLDS] \
        + [f"mean_conf_ge{int(round(t*100)):02d}" for t in THRESHOLDS] \
        + [f"top{k}_mean_conf" for k in TOPKS]

    header = ["stem"]
    for path_tag in ("super", "base"):
        for lvl in LEVELS:
            for s in stat_names:
                header.append(f"{s}_{lvl}_{path_tag}")
    header += ["n_gt_small", "n_gt_medium", "n_gt_large", "n_gt_total", "gt_med_area"]

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))

    with torch.no_grad():
        pbar = tqdm(loader, total=n)
        for i, batch in enumerate(pbar):
            if args.limit and i >= args.limit: break
            batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255.0
            img_h, img_w = batch["img"].shape[-2], batch["img"].shape[-1]

            preds_super = model.predict(batch["img"], skip=skip_super)
            preds_base  = model.predict(batch["img"], skip=skip_base)
            # predict() returns (decoded, per_level_list) for non-export inference.
            raw_levels_super = preds_super[1] if isinstance(preds_super, (list, tuple)) else None
            raw_levels_base  = preds_base[1]  if isinstance(preds_base,  (list, tuple)) else None
            if raw_levels_super is None or raw_levels_base is None:
                raise SystemExit("Could not retrieve per-level raw tensors from predict().")

            super_stats = [per_level_conf_stats(t, reg_max, nc) for t in raw_levels_super]
            base_stats  = [per_level_conf_stats(t, reg_max, nc) for t in raw_levels_base]

            stem = Path(batch["im_file"][0]).stem
            row = [stem]
            for stats_list in (super_stats, base_stats):
                for lvl_stats in stats_list:
                    row += [lvl_stats[s] for s in stat_names]

            gt = gt_area_bins(batch["bboxes"], img_h, img_w)
            row += [gt["n_gt_small"], gt["n_gt_medium"], gt["n_gt_large"],
                    gt["n_gt_total"], gt["gt_med_area"]]
            rows.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[*] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
