"""
Finetune AnyDepth-YOLOv12 on KITTI 2D Object Detection (7 classes).

The detection head from a BDD100K-trained checkpoint has 10 outputs while
KITTI here has 7 classes; ultralytics handles the mismatch automatically by
loading only matched layers (backbone + neck) and re-initializing the head.

Hyper-parameters tuned for short domain-adaptation finetuning on a small
dataset (~6K train images):
  - 20 epochs (short — finetuning, not from scratch)
  - lr0 = 1e-4 (10x lower than BDD finetune; pretrained backbone preserved)
  - imgsz 1248 (32-multiple of KITTI native 1242 width, rect padding for 384h)
  - SGD, momentum 0.9, mosaic OFF, mild flip
  - batch 16/32 depending on GPU memory
"""
from ultralytics import YOLO
import argparse
from ultralytics.utils import RANK

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--imgsz', type=int, default=1248)
parser.add_argument('--project', type=str, default='')
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--task', type=str, default='detect')
parser.add_argument('--data', type=str, default='kitti.yaml')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr0', type=float, default=1e-4)
args = parser.parse_args()

IMG_SIZE = args.imgsz
weight_path = args.weight.strip()

if RANK in {-1, 0}:
    print(f"[*] Image size: {IMG_SIZE}")

if args.resume:
    if not weight_path:
        raise ValueError("--resume requires --weight to point to a checkpoint file")
    if RANK in {-1, 0}:
        print(f"[*] Resuming training from {weight_path}")
    model = YOLO(weight_path, task=args.task)
    results = model.train(resume=True, rect=True)

else:
    if not args.config:
        raise ValueError("--config is required when starting a new training run")
    if RANK in {-1, 0}:
        if weight_path:
            print(f"[*] Starting KITTI finetune from pretrained: {weight_path}")
        else:
            print("[*] Starting new scratch training from config (no --weight)")

    model = YOLO(args.config, task=args.task)
    if weight_path:
        # Loads backbone+neck; mismatched head (10->7) is auto re-initialized.
        model.load(weight_path)

    results = model.train(
        task=args.task,
        project=args.project,
        data=args.data,

        # short finetuning schedule
        epochs=args.epoch,
        optimizer='SGD',
        momentum=0.900,
        batch=args.batch,         # KITTI is small; 16 fits most GPUs at 1248
        nbs=64,                   # effective batch for lr scaling
        lr0=args.lr0,             # 1e-4: small LR for finetune on small data
        lrf=1e-2,                 # final = lr0 * lrf = 1e-6
        warmup_epochs=2.0,

        # image
        imgsz=IMG_SIZE,
        rect=True,                # KITTI ~1242x375 (very wide aspect)

        # augmentation — keep tiny for finetune
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,               # horizontal flip OK for driving scenes
        flipud=0.0,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,

        # reproducibility
        seed=args.seed,
        deterministic=True,

        # device="0,1",
        device="0",
    )


'''
# AnyDepth finetune on KITTI (from BDD-finetuned weight)
mkdir -p ./runs/kitti/detect/anydepth-yolov12s
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node 2 train_adn_kitti.py \
  --task detect \
  --config ./ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
  --data kitti.yaml \
  --epoch 20 \
  --imgsz 1248 \
  --batch 16 \
  --lr0 1e-4 \
  --weight ./pretrained_coco/yolo-ad-v12-orig_small_0.479_0.451.pt \
  --project ./runs/kitti/detect/anydepth-yolov12s \
  2>&1 | tee ./runs/kitti/detect/anydepth-yolov12s/train.log
  
mkdir -p ./runs/kitti/detect/anydepth-yolov12s
export CUDA_VISIBLE_DEVICES=0
python train_adn_kitti.py \
  --task detect \
  --config ./ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
  --data kitti.yaml \
  --epoch 20 \
  --imgsz 1248 \
  --batch 16 \
  --lr0 1e-4 \
  --weight ./pretrained_coco/yolo-ad-v12-orig_small_0.479_0.451.pt \
  --project ./runs/kitti/detect/anydepth-yolov12s \
  2>&1 | tee ./runs/kitti/detect/anydepth-yolov12s/train.log  
'''
