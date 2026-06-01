"""Offline cache for method01 policy training.

Because the detector is frozen, everything the policy needs is a fixed function
of each image and is precomputed once here:

  per image i, for path p in {BASE, SUPER}:
    - input_vec[p] : concat(GAP of backbone layers 4,6,8)     [Ci]
    - pred_vec[p]  : concat(GAP of neck   layers 14,17,20)    [Cp]
    - loss[p]      : per-image detection loss (v8DetectionLoss vs GT boxes)

Saved to a single .pt so policy training reads tensors only (no detector
forward), making epochs trivially cheap and lambda sweeps fast.

Usage:
    python method01_advantage_regress/build_cache.py \
        --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
        --data ultralytics/cfg/datasets/kitti.yaml --split train \
        --imgsz 384 1248 --batch 16 \
        --out runs/kitti/policy/cache_train.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

from method01_advantage_regress.feature_tap import STATE_LAYERS, split_features

SKIPPABLE_TYPES = None  # resolved lazily from tasks.py


def num_skippable(model):
    from ultralytics.nn.tasks import (
        SkippableC3k2, SwitchableC3k2, SwitchableC2f, SwitchableA2C2f, SkippableA2C2f,
    )
    return sum(
        isinstance(m, (SkippableC3k2, SwitchableC3k2, SwitchableC2f, SwitchableA2C2f, SkippableA2C2f))
        for m in model.model
    )


@torch.no_grad()
def forward_path(model, img, skip, detect):
    """eval-BN forward with training-format Detect output + GAP features."""
    detect.training = True  # raw feature maps for the loss, BN stays in eval
    out = model._predict_once(img, False, False, None, skip=skip, return_features=True)
    detect.training = False
    return out["pred"], out["features"]


def per_image_losses(criterion, preds, batch):
    """Detection loss for each image in the batch by slicing targets per index."""
    B = preds[0].shape[0]
    losses = []
    for i in range(B):
        sub = {
            "img": batch["img"][i:i + 1],
            "batch_idx": torch.zeros((batch["batch_idx"] == i).sum(), device=preds[0].device),
            "cls": batch["cls"][batch["batch_idx"] == i],
            "bboxes": batch["bboxes"][batch["batch_idx"] == i],
        }
        preds_i = [p[i:i + 1] for p in preds]
        loss_sum, _ = criterion(preds_i, sub)
        losses.append(loss_sum.detach().sum())
    return torch.stack(losses)  # [B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt")
    ap.add_argument("--data", default="ultralytics/cfg/datasets/kitti.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1248], help="H W")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="kitti", help="output scope: outputs/<dataset>/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(Path(__file__).resolve().parent / "outputs" / args.dataset / f"cache_{args.split}.pt")

    device = args.device if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.weight, task="detect")
    model = yolo.model.to(device).eval()
    model.save = sorted(set(list(model.save) + list(STATE_LAYERS)))
    model.num_skippable_layers = num_skippable(model)
    N = model.num_skippable_layers
    detect = model.model[-1]
    criterion = v8DetectionLoss(model)
    # ensure loss gains are attribute-accessible (model.args may be a dict)
    if isinstance(getattr(criterion, "hyp", None), dict) or not hasattr(criterion.hyp, "box"):
        criterion.hyp = DEFAULT_CFG

    # dataloader (rect, no augmentation -- deterministic features)
    data = check_det_dataset(args.data)
    cfg = DEFAULT_CFG
    cfg.imgsz = max(args.imgsz)
    cfg.rect = True
    dataset = build_yolo_dataset(cfg, data[args.split], args.batch, data,
                                 mode="val", rect=True, stride=int(model.stride.max()))
    loader = build_dataloader(dataset, args.batch, workers=4, shuffle=False)

    skip_base = [True] * N
    skip_super = [False] * N

    rows = {"input_base": [], "pred_base": [], "input_super": [], "pred_super": [],
            "loss_base": [], "loss_super": [], "im_file": []}

    for bi, batch in enumerate(loader):
        img = batch["img"].to(device).float() / 255.0
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["img"] = img

        for skip, tag in ((skip_base, "base"), (skip_super, "super")):
            preds, feats = forward_path(model, img, skip, detect)
            inp, prd = split_features(feats, model.save)
            rows[f"input_{tag}"].append(torch.cat(inp, dim=1).cpu())
            rows[f"pred_{tag}"].append(torch.cat(prd, dim=1).cpu())
            rows[f"loss_{tag}"].append(per_image_losses(criterion, preds, batch).cpu())

        rows["im_file"].extend(batch["im_file"])
        if bi % 20 == 0:
            print(f"[{bi}] cached {sum(t.shape[0] for t in rows['loss_base'])} images")

    cache = {k: (torch.cat(v) if torch.is_tensor(v[0]) else v) for k, v in rows.items()}
    cache["meta"] = {"weight": str(args.weight), "split": args.split,
                     "imgsz": list(args.imgsz), "num_skippable": N}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    n = cache["loss_base"].shape[0]
    print(f"[*] cached {n} images -> {out}")
    print(f"[*] input_vec dim={cache['input_base'].shape[1]}, pred_vec dim={cache['pred_base'].shape[1]}")
    print(f"[*] mean L_base={cache['loss_base'].mean():.3f}, mean L_super={cache['loss_super'].mean():.3f}, "
          f"mean A={(cache['loss_base']-cache['loss_super']).mean():.4f}")


if __name__ == "__main__":
    main()
