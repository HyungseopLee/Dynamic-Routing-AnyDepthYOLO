"""
AnyDepth Step 2 — train the AdaptiveRouter on top of a frozen SUPER/BASE cascaded detector.

Defaults mirror DynamicDet/README.md:
    --epoch 2  --adam  --batch-size 1  --imgsz 640
  (router is a tiny 2-layer 1x1 conv; detector forward twice uses no_grad;
   long training gives no additional benefit.)

Usage:
    python train_step2.py \
        --config ./ultralytics/cfg/models/v12/yolo-ad-v12.yaml \
        --weight ./runs/bdd100k/detect/anydepth-yolov12l/.../weights/best.pt \
        --data   bdd100k.yaml \
        --epoch  2 \
        --imgsz  1280 \
        --project ./runs/bdd100k/step2/anydepth-yolov12l

Notes:
    - Protocol mirrors DynamicDet/train_step2.py: batch=1 (DynamicDet enforces this).
    - Only `router` params are updated; SUPER/BASE detector weights stay frozen.
"""
import argparse

from ultralytics import YOLO
from ultralytics.utils import RANK
from ultralytics.data.utils import check_det_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True,
                   help='Model YAML (e.g., yolo-ad-v12.yaml)')
    p.add_argument('--weight', type=str, required=True,
                   help='Pretrained Step 1 checkpoint (cascaded detector)')
    p.add_argument('--data', type=str, required=True, help='Dataset yaml')
    p.add_argument('--project', type=str, required=True)
    p.add_argument('--epoch', type=int, default=2,
                   help='DynamicDet uses 2 epochs; router converges fast')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--optimizer', type=str, default='AdamW',
                   help='DynamicDet uses --adam (AdamW); SGD also supported')
    # Defaults below copied from DynamicDet/hyp/hyp.finetune.dynamic.adam.yaml
    p.add_argument('--lr0', type=float, default=1e-5,
                   help='DynamicDet: 1e-5 (constant, router-only fine-tune)')
    p.add_argument('--lrf', type=float, default=1.0,
                   help='DynamicDet: 1.0 -> constant LR')
    p.add_argument('--weight-decay', type=float, default=0.005,
                   help='DynamicDet: 0.005')
    p.add_argument('--warmup-epochs', type=float, default=0.01,
                   help='DynamicDet: 0.01')
    p.add_argument('--warmup-bias-lr', type=float, default=0.01,
                   help='DynamicDet: 0.01')
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--name', type=str, default='step2_router')
    return p.parse_args()


def main():
    args = parse_args()

    if RANK in {-1, 0}:
        print(f"[Step2] config={args.config}  weight={args.weight}")
        print(f"[Step2] data={args.data}  imgsz={args.imgsz}  epoch={args.epoch}  "
              f"optimizer={args.optimizer}")

    # Resolve dataset nc BEFORE building the model so the detection head
    # is created with the correct class count (best.pt was trained on nc=10
    # for BDD100K; the YAML default is nc=80 which would cause a head
    # shape mismatch when loading weights).
    data_dict = check_det_dataset(args.data)
    nc = int(data_dict["nc"])
    if RANK in {-1, 0}:
        print(f"[Step2] dataset nc={nc} (from {args.data})")

    # Build model with correct nc, then load pretrained weights
    model = YOLO(args.weight, task='detect_step2')
    # Rebuild the underlying DetectionModelAnyDepth with the right nc
    from ultralytics.nn.tasks import DetectionModelAnyDepth
    model.model = DetectionModelAnyDepth(args.config, nc=nc, verbose=RANK == -1)
    # model.overrides["nc"] = nc
    # model.load(args.weight)

    # All hyperparameters mirror DynamicDet/hyp/hyp.finetune.dynamic.adam.yaml
    model.train(
        task='detect_step2',
        data=args.data,
        project=args.project,
        name=args.name,

        epochs=args.epoch,
        optimizer=args.optimizer,
        momentum=0.937,          # DynamicDet: 0.937 (Adam beta1)
        batch=1,                 # DynamicDet: batch_size=1 (enforced)
        nbs=1,                   # accumulate every step (no grad accumulation)
        lr0=args.lr0,            # DynamicDet: 1e-5
        lrf=args.lrf,            # DynamicDet: 1.0 (constant LR)
        weight_decay=args.weight_decay,       # DynamicDet: 0.005
        warmup_epochs=args.warmup_epochs,     # DynamicDet: 0.01
        warmup_momentum=0.8,                  # DynamicDet: 0.8
        warmup_bias_lr=args.warmup_bias_lr,   # DynamicDet: 0.01

        imgsz=args.imgsz,
        rect=True,

        # Augmentations copied from hyp.finetune.dynamic.adam.yaml
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.2, scale=0.9,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        close_mosaic=0,

        # Loss gains (DynamicDet: box 0.05, cls 0.3, obj 0.7 — YOLOv12 has no obj)
        box=0.05, cls=0.3,

        device=args.device,
    )


if __name__ == '__main__':
    main()


'''
    # --config ./ultralytics/cfg/models/v12/yolo-ad-v12l.yaml \
        
# Example — BDD100K
mkdir -p ./runs/bdd100k/step2/anydepth-yolov12l
python train_step2.py \
    --weight ./runs/bdd100k/detect/anydepth-yolov12l/_30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-740_singleScale_augNothing/weights/best.pt \
    --data   bdd100k.yaml \
    --epoch  2 \
    --imgsz  1280 \
    --optimizer AdamW \
    --project ./runs/bdd100k/step2/anydepth-yolov12l \
    --device 0 \
    2>&1 | tee ./runs/bdd100k/step2/anydepth-yolov12l/step2.log
'''
