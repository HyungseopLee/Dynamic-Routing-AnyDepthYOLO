"""Finetune the AnyDepth-YOLOv12s detector on Waymo (FRONT camera) at native
1920x1280, starting from the BDD anydepth checkpoint. The detection head is
reinitialised for Waymo's 4-class taxonomy (vehicle/pedestrian/sign/cyclist);
the backbone + neck weights transfer from BDD. Self-distillation alpha_base=0.2
is kept so the per-image base/super advantage stays wide for the router.

Native resolution is preserved by imgsz=1920 (long side) + rect=True: Waymo FRONT
is 1920x1280 (3:2), both multiples of stride 32, so frames train un-padded at
1920x1280 (vs BDD's 1280x720).

    export CUDA_VISIBLE_DEVICES=0,1
    python -m torch.distributed.run --nproc_per_node 2 train_adn_waymo.py \
        --config ./ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
        --weight finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.2_orig_mAP35.1_33.8.pt \
        --data waymo.yaml --epoch 30 --imgsz 1920 --batch 16 --alpha_base 0.2 --lr0 1e-3 \
        --project ./runs/waymo/detect/anydepth-yolov12s \
        2>&1 | tee ./runs/waymo/detect/anydepth-yolov12s/30e_SGD0900_bs16_nbs256_1e-3_1e-5_1920-1280_singleScale_augNothing_alpha0.2_orig.log
"""
import argparse

from ultralytics import YOLO
from ultralytics.utils import RANK

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='./ultralytics/cfg/models/v12/yolo-ad-v12-orig.yaml',
                    help='anydepth architecture cfg (scale s auto-assigned)')
parser.add_argument('--weight', type=str, required=True,
                    help='pretrained weights to load into the cfg (COCO anydepth-s .pt)')
parser.add_argument('--data', type=str, default='waymo.yaml')
parser.add_argument('--task', type=str, default='detect')
parser.add_argument('--project', type=str, default='./runs/waymo/detect/anydepth-yolov12s')
parser.add_argument('--name', type=str, default='',
                    help='run subfolder name under --project (ultralytics save dir)')
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--imgsz', type=int, default=1920, help='long side; 1920 keeps Waymo native')
parser.add_argument('--batch', type=int, default=16, help='global batch (lower for 1920x1280 memory)')
parser.add_argument('--lr0', type=float, default=1e-3, help='finetune LR')
parser.add_argument('--alpha_base', type=float, default=0.2,
                    help='self-distillation weight for the base path (router needs ~0.2)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='0,1')
args = parser.parse_args()

if RANK in {-1, 0}:
    print(f"[*] Finetune AnyDepth on Waymo  cfg={args.config}  weight={args.weight}")
    print(f"[*] imgsz={args.imgsz} (rect, native 1920x1280)  batch={args.batch}  alpha_base={args.alpha_base}")

# Build the anydepth-s architecture from cfg, then load COCO-pretrained weights
# (head is re-fit to Waymo's 3 classes from --data during train()).
model = YOLO(args.config, task=args.task)
model.load(args.weight)

results = model.train(
    task=args.task,
    project=args.project,
    name=args.name or None,
    data=args.data,

    epochs=args.epoch,
    optimizer='SGD',
    momentum=0.900,
    batch=args.batch,
    nbs=256,            # effective batch via gradient accumulation
    lr0=args.lr0,
    lrf=1e-2,

    imgsz=args.imgsz,
    rect=True,          # keep 3:2 aspect -> native 1920x1280, no square pad

    # no aug (match BDD recipe)
    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,
    copy_paste=0.0,
    flipud=0.0,

    alpha_base=args.alpha_base,

    seed=args.seed,
    deterministic=True,
    device=args.device,
)
