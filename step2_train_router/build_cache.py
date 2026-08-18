"""Offline feature cache for router training.

Because the detector is frozen, everything the router needs is a fixed function
of each image and is precomputed once here:

  per image i, for path p in {BASE, SUPER}:
    - input_vec[p] : concat(GAP of backbone layers 4,6,8)     [Ci]
    - pred_vec[p]  : concat(GAP of neck   layers 14,17,20)    [Cp]
    - loss[p]      : per-image detection loss (v8DetectionLoss vs GT boxes)

Saved to a single .pt so router training reads tensors only (no detector
forward), making epochs trivially cheap and lambda sweeps fast.

Usage (run from repo root):
    # KITTI  (384x1248)
    python -m step2_train_router.build_cache \
        --weight results/step1_finetune/weights/kitti/best.pt \
        --data ultralytics/cfg/datasets/kitti.yaml --dataset kitti \
        --split train --imgsz 384 1248 --batch 16

    # BDD100K  (720x1280; --feat both for TinyConv router)
    python -m step2_train_router.build_cache \
        --weight results/step1_finetune/weights/bdd100k/best.pt \
        --data ultralytics/cfg/datasets/bdd100k.yaml --dataset bdd100k \
        --split train --imgsz 720 1280 --feat both --batch 16
    python -m step2_train_router.build_cache \
        --weight results/step1_finetune/weights/bdd100k/best.pt \
        --data ultralytics/cfg/datasets/bdd100k.yaml --dataset bdd100k \
        --split val --imgsz 720 1280 --feat both --batch 16

    # Waymo  (1280x1920; large -- use --fp16 to halve disk usage)
    python -m step2_train_router.build_cache \
        --weight results/step1_finetune/weights/waymo/best.pt \
        --data ultralytics/cfg/datasets/waymo.yaml --dataset waymo \
        --split train --imgsz 1280 1920 --feat both --fp16 --batch 16
    python -m step2_train_router.build_cache \
        --weight results/step1_finetune/weights/waymo/best.pt \
        --data ultralytics/cfg/datasets/waymo.yaml --dataset waymo \
        --split val --imgsz 1280 1920 --feat both --fp16 --batch 16
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

import torch.nn.functional as F

from router.feature_tap import (
    STATE_LAYERS, INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS,
)

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
    """eval-BN forward returning training-format Detect output (for the loss).
    Raw spatial feature maps are captured by hooks (see main)."""
    detect.training = True  # raw feature maps for the loss, BN stays in eval
    out = model._predict_once(img, False, False, None, skip=skip, return_features=True)
    detect.training = False
    return out["pred"]


def parse_grid(s):
    """'G' -> (G,G) square; 'HxW' -> (H,W) rectangular (aspect-preserving)."""
    if "x" in str(s):
        h, w = str(s).split("x"); return (int(h), int(w))
    return (int(s), int(s))


def grids(captured, layers, G):
    """Concat adaptive-pooled (G=(Gh,Gw)) spatial maps of `layers` along channels
    -> [B,sumC,Gh,Gw]."""
    return torch.cat([F.adaptive_avg_pool2d(captured[l].float(), G) for l in layers], dim=1)


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
    ap.add_argument("--weight", default="results/step1_finetune/weights/kitti/best.pt")
    ap.add_argument("--data", default="ultralytics/cfg/datasets/kitti.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1248], help="H W")
    ap.add_argument("--base_imgsz", type=int, nargs=2, default=None,
                    help="H W for the BASE path only (resolution routing). If set, the base "
                         "path runs on a bilinearly-downsampled image (must be stride-32 mult); "
                         "the super path keeps --imgsz. Default: same as --imgsz (depth-only).")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="kitti", help="output scope: outputs/<dataset>/")
    ap.add_argument("--grid", default="4", help="spatial grid: 'G' (square GxG) or 'HxW' "
                    "(rectangular, e.g. '2x6' to preserve KITTI aspect ratio)")
    ap.add_argument("--feat", default="both", choices=["input", "pred", "both"],
                    help="which feature group(s) to cache; 'input' (backbone-only) skips "
                         "caching the neck/pred grids to roughly halve cache size")
    ap.add_argument("--fp16", action="store_true",
                    help="store feature grids as float16 to halve RAM/disk usage")
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
    G = parse_grid(args.grid)

    # forward hooks capture raw spatial maps [B,C,H,W] for the tapped layers
    captured = {}
    handles = [model.model[l].register_forward_hook(
        lambda m, i, o, l=l: captured.__setitem__(l, o)) for l in STATE_LAYERS]

    cache_input = args.feat in ("input", "both")
    cache_pred = args.feat in ("pred", "both")

    # chunk_save: flush every N batches to a temp file to bound peak RAM usage.
    # Final pass concatenates chunks. Activated when --fp16 is set (large grids).
    CHUNK = 200  # batches per flush (~200*8=1600 images)
    chunk_dir = Path(args.out).parent / f"_chunks_{Path(args.out).stem}"
    use_chunks = args.fp16

    if use_chunks:
        import shutil
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True)

    rows = {"loss_base": [], "loss_super": [], "im_file": []}
    for tag in ("base", "super"):
        if cache_input: rows[f"input_{tag}"] = []
        if cache_pred: rows[f"pred_{tag}"] = []

    chunk_idx = 0

    def flush_chunk():
        nonlocal chunk_idx, rows
        feat_keys = {"input_base", "input_super", "pred_base", "pred_super"}
        chunk = {}
        for k, v in rows.items():
            if v and torch.is_tensor(v[0]):
                t = torch.cat(v).half() if k in feat_keys else torch.cat(v)
                chunk[k] = t
            else:
                chunk[k] = list(v)
        torch.save(chunk, chunk_dir / f"chunk_{chunk_idx:04d}.pt")
        chunk_idx += 1
        rows = {"loss_base": [], "loss_super": [], "im_file": []}
        for tag in ("base", "super"):
            if cache_input: rows[f"input_{tag}"] = []
            if cache_pred: rows[f"pred_{tag}"] = []

    for bi, batch in enumerate(loader):
        img = batch["img"].to(device).float() / 255.0
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["img"] = img

        for skip, tag in ((skip_base, "base"), (skip_super, "super")):
            captured.clear()
            if tag == "base" and args.base_imgsz is not None:
                img_t = F.interpolate(img, size=tuple(args.base_imgsz), mode="bilinear", align_corners=False)
                batch_t = {**batch, "img": img_t}
            else:
                img_t, batch_t = img, batch
            preds = forward_path(model, img_t, skip, detect)
            if cache_input:
                rows[f"input_{tag}"].append(grids(captured, INPUT_LEVEL_LAYERS, G).cpu())
            if cache_pred:
                rows[f"pred_{tag}"].append(grids(captured, PRED_LEVEL_LAYERS, G).cpu())
            rows[f"loss_{tag}"].append(per_image_losses(criterion, preds, batch_t).cpu())

        rows["im_file"].extend(batch["im_file"])
        if bi % 20 == 0:
            n = sum(t.shape[0] for t in rows["loss_base"])
            print(f"[{bi}] cached {n} images (chunk {chunk_idx})")
        if use_chunks and (bi + 1) % CHUNK == 0:
            flush_chunk()

    if use_chunks and rows["loss_base"]:
        flush_chunk()

    for h in handles:
        h.remove()

    feat_keys = {"input_base", "input_super", "pred_base", "pred_super"}
    if use_chunks:
        # merge chunks
        print(f"[*] merging {chunk_idx} chunks ...")
        parts = {k: [] for k in (list(feat_keys) + ["loss_base", "loss_super", "im_file"])}
        for ci in range(chunk_idx):
            ch = torch.load(chunk_dir / f"chunk_{ci:04d}.pt", map_location="cpu", weights_only=False)
            for k in parts:
                if k in ch:
                    parts[k].append(ch[k])
        cache = {}
        for k, vlist in parts.items():
            if vlist and torch.is_tensor(vlist[0]):
                cache[k] = torch.cat(vlist)
            elif vlist:
                cache[k] = [x for sub in vlist for x in sub]
        import shutil; shutil.rmtree(chunk_dir)
    else:
        cache = {}
        for k, v in rows.items():
            if v and torch.is_tensor(v[0]):
                t = torch.cat(v)
                if args.fp16 and k in feat_keys:
                    t = t.half()
                cache[k] = t
            else:
                cache[k] = v
    cache["meta"] = {"weight": str(args.weight), "split": args.split,
                     "imgsz": list(args.imgsz), "num_skippable": N, "grid": G,
                     "feat": args.feat, "base_imgsz": args.base_imgsz}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    n = cache["loss_base"].shape[0]
    print(f"[*] cached {n} images (feat={args.feat}) -> {out}")
    if cache_input:
        print(f"[*] input grid={tuple(cache['input_base'].shape[1:])}")
    if cache_pred:
        print(f"[*] pred grid={tuple(cache['pred_base'].shape[1:])}")
    print(f"[*] mean L_base={cache['loss_base'].mean():.3f}, mean L_super={cache['loss_super'].mean():.3f}, "
          f"mean A={(cache['loss_base']-cache['loss_super']).mean():.4f}")


if __name__ == "__main__":
    main()
