"""
Per-image detection loss on BDD100K using an AnyDepth model.

Runs both Super-net (skip=[False]*N) and Base-net (skip=[True]*N) on each image,
computes (box, cls, dfl) via v8DetectionLoss, and writes a CSV:
  stem,
  loss_super, box_super, cls_super, dfl_super,
  loss_base,  box_base,  cls_base,  dfl_base,
  loss_diff,                                       # base - super (>0 => base struggles more)
  n_gt

Usage:
  PYTHONPATH=$PWD \
  python tools/per_image_loss.py \
    --weight ./finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.6_orig_mAP34.3_33.1.pt \
    --data bdd100k.yaml \
    --imgsz 1280 \
    --out runs/loss_analysis/per_image_loss.csv \
    2>&1 | tee ./runs/loss_analysis/per_image_loss.log
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--data", default="bdd100k.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="runs/loss_analysis/per_image_loss.csv")
    p.add_argument("--limit", type=int, default=0, help="debug: cap #images")
    return p.parse_args()


def loss_items_to_tuple(loss_items):
    box, cls_l, dfl = [float(x) for x in loss_items.detach().cpu().tolist()]
    return box, cls_l, dfl, box + cls_l + dfl


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
    dataset = build_yolo_dataset(cfg, img_path, batch=1, data=data, mode="val", stride=int(max(model.stride)), rect=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    )

    # Build v8DetectionLoss directly to avoid AnyDepth KD-loss requirements.
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils import IterableSimpleNamespace
    merged_args = dict(DEFAULT_CFG_DICT)
    ckpt_args = model.args if isinstance(model.args, dict) else vars(model.args)
    merged_args.update(ckpt_args)
    model.args = IterableSimpleNamespace(**merged_args)
    criterion = v8DetectionLoss(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip_super = [False] * num_skip
    skip_base  = [True]  * num_skip

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

            preds_super = model.predict(batch["img"], skip=skip_super)
            _, items_super = criterion(preds_super, batch)
            box_s, cls_s, dfl_s, total_s = loss_items_to_tuple(items_super)

            preds_base = model.predict(batch["img"], skip=skip_base)
            _, items_base = criterion(preds_base, batch)
            box_b, cls_b, dfl_b, total_b = loss_items_to_tuple(items_base)

            stem = Path(batch["im_file"][0]).stem
            n_gt = int(batch["cls"].numel())
            rows.append((stem,
                         total_s, box_s, cls_s, dfl_s,
                         total_b, box_b, cls_b, dfl_b,
                         total_b - total_s, n_gt))
            if i % 100 == 0:
                pbar.set_postfix(s=f"{total_s:.2f}", b=f"{total_b:.2f}")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem",
                    "loss_super", "box_super", "cls_super", "dfl_super",
                    "loss_base",  "box_base",  "cls_base",  "dfl_base",
                    "loss_diff", "n_gt"])
        w.writerows(rows)
    print(f"[*] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
