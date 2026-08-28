"""Finetune AnyDepth-YOLOv12 on a target dataset.

The detector backbone + neck weights transfer from a pretrained checkpoint;
the detection head is re-initialised automatically when the number of classes
differs (e.g. BDD→KITTI or BDD→Waymo). Keep --alpha_base 0.2 so the
per-image base/super advantage stays wide enough for the router.

Usage (run from repo root):

    # BDD100K  (720x1280, 30 epochs, 2 GPUs)
    export CUDA_VISIBLE_DEVICES=0,1
    python -m torch.distributed.run --nproc_per_node 2 step1_finetune/finetune.py \
        --config ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
        --weight results/step1_finetune/pretrained_coco/yolo-ad-v12-orig_small_0.479_0.451.pt \
        --data ultralytics/cfg/datasets/bdd100k.yaml \
        --dataset bdd100k --epoch 30 --imgsz 1280 --batch 32 --lr0 1e-3

    # KITTI  (384x1248, 20 epochs, 1 GPU — small dataset)
    export CUDA_VISIBLE_DEVICES=0
    python step1_finetune/finetune.py \
        --config ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
        --weight results/step1_finetune/weights/bdd100k/best.pt \
        --data ultralytics/cfg/datasets/kitti.yaml \
        --dataset kitti --epoch 20 --imgsz 1248 --batch 16 --lr0 1e-4

    # Waymo  (1280x1920, 30 epochs, 2 GPUs — native resolution)
    export CUDA_VISIBLE_DEVICES=0,1
    python -m torch.distributed.run --nproc_per_node 2 step1_finetune/finetune.py \
        --config ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
        --weight results/step1_finetune/weights/bdd100k/best.pt \
        --data ultralytics/cfg/datasets/waymo.yaml \
        --dataset waymo --epoch 30 --imgsz 1920 --batch 16 --lr0 1e-3

Checkpoints are saved under results/step1_finetune/logs/<dataset>/detect/<run_name>/.
"""
import argparse

from ultralytics import YOLO
from ultralytics.utils import RANK

# Dataset-specific defaults (overridden by explicit CLI args)
DATASET_DEFAULTS = {
    "bdd100k": dict(imgsz=1280, epoch=30, lr0=1e-3, batch=32, nbs=256, alpha_base=0.2),
    "kitti":   dict(imgsz=1248, epoch=20, lr0=1e-4, batch=16, nbs=64,  alpha_base=0.2),
    "waymo":   dict(imgsz=1920, epoch=30, lr0=1e-3, batch=16, nbs=256, alpha_base=0.2),
}

ap = argparse.ArgumentParser()
ap.add_argument("--config",  required=True,  help="AnyDepth model yaml (e.g. yolo-ad-v12s-orig.yaml)")
ap.add_argument("--weight",  default="",     help="Pretrained .pt to load; leave empty for scratch")
ap.add_argument("--data",    required=True,  help="Dataset yaml path")
ap.add_argument("--dataset", default="",     help="One of {bdd100k,kitti,waymo} — loads default hparams")
ap.add_argument("--task",    default="detect")
ap.add_argument("--project", default="",     help="Output dir (default: results/step1_finetune/logs/<dataset>/detect)")
ap.add_argument("--epoch",       type=int,   default=None)
ap.add_argument("--imgsz",       type=int,   default=None)
ap.add_argument("--batch",       type=int,   default=None)
ap.add_argument("--nbs",         type=int,   default=None,  help="Nominal batch size for LR scaling")
ap.add_argument("--lr0",         type=float, default=None)
ap.add_argument("--alpha_base",  type=float, default=None,
                help="Self-distillation weight for base path; ~0.2 widens base/super advantage for routing")
ap.add_argument("--device",  default="0,1")
ap.add_argument("--seed",    type=int, default=0)
ap.add_argument("--resume",  action="store_true")
args = ap.parse_args()

# Apply dataset defaults, then override with any explicit CLI arg
defs = DATASET_DEFAULTS.get(args.dataset, {})
epoch      = args.epoch      if args.epoch      is not None else defs.get("epoch",      30)
imgsz      = args.imgsz      if args.imgsz      is not None else defs.get("imgsz",      1280)
batch      = args.batch      if args.batch      is not None else defs.get("batch",      32)
nbs        = args.nbs        if args.nbs        is not None else defs.get("nbs",        256)
lr0        = args.lr0        if args.lr0        is not None else defs.get("lr0",        1e-3)
alpha_base = args.alpha_base if args.alpha_base is not None else defs.get("alpha_base", 0.2)
project    = args.project or f"results/step1_finetune/logs/{args.dataset or 'custom'}/detect"

if RANK in {-1, 0}:
    print(f"[*] dataset={args.dataset}  imgsz={imgsz}  epoch={epoch}  "
          f"batch={batch}  lr0={lr0}  alpha_base={alpha_base}")

if args.resume:
    if not args.weight:
        raise ValueError("--resume requires --weight pointing to a checkpoint")
    model = YOLO(args.weight, task=args.task)
    model.train(resume=True, rect=True)
else:
    model = YOLO(args.config, task=args.task)
    if args.weight:
        model.load(args.weight)

    model.train(
        task=args.task,
        project=project,
        data=args.data,

        epochs=epoch,
        optimizer="SGD",
        momentum=0.900,
        batch=batch,
        nbs=nbs,
        lr0=lr0,
        lrf=1e-2,

        imgsz=imgsz,
        rect=True,

        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        flipud=0.0,

        alpha_base=alpha_base,

        seed=args.seed,
        deterministic=True,
        device=args.device,
    )
