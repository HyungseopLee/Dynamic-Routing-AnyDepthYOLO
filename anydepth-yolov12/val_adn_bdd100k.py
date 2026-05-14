from ultralytics import YOLO 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', type=int, default=640, help='Image size (long side, e.g., 640 or 1280)')
parser.add_argument('--project', type=str, default='')
parser.add_argument('--weight', type=str, default='')
args = parser.parse_args()
IMG_SIZE = args.imgsz

model = YOLO(args.weight, task='detect')
num_skip = getattr(model.model, 'num_skippable_layers', 0)
is_anydepth = num_skip > 0
print(f"model: {model}")
print(f"[*] AnyDepth model: {is_anydepth}, num_skippable_layers: {num_skip}")
print(f"[*] Running validation with Image Size: {IMG_SIZE}")


if is_anydepth:
    # Full
    skip = [False,] * model.num_skippable_layers
    model.val(
        task='detect',
        data='bdd100k.yaml', 
        save_json=True, 
        skip=skip,
        project=args.project,
        name=f'Full_{IMG_SIZE}',
        
        imgsz=IMG_SIZE,
        rect=True, # for IMG_SIZE x IMG_SIZE
    )
    # Base
    skip = [True,] * model.num_skippable_layers
    model.val(
        task='detect',
        data='bdd100k.yaml', 
        save_json=True, 
        skip=skip,
        project=args.project,
        name=f'Base_{IMG_SIZE}',
        
        imgsz=IMG_SIZE,
        rect=True,
    )

else:
    model.val(
        task='detect',
        data='bdd100k.yaml', 
        save_json=True, 
        project=args.project,
        name=f'Baseline_{IMG_SIZE}',
        
        imgsz=IMG_SIZE,
        rect=True,
    )



'''

# weight path
(bdd100k) baseline: ./runs/bdd100k/detect/baseline-yolov12l/train_50e_SGD0900_bs64_nbs256_imgsz640_scale050/weights/best.pt
(bdd100k) any-depth: ./runs/bdd100k/detect/anydepth-yolov12l/train_50e_SGD0900_bs64_nbs256_imgsz640_scale050/weights/best.pt


# baseline
## val on 1-gpu
mkdir -p ./runs/bdd100k/detect/baseline-yolov12s
python val_adn_bdd100k.py \
    --imgsz 1280 \
    --weight /home/hslee/context-anydepth-det/anydepth-yolov12/runs/bdd100k/detect/anydepth-yolov12s/train/weights/best.pt \
    --project ./runs/bdd100k/detect/baseline-yolov12s/train \
    2>&1 | tee ./runs/bdd100k/detect/baseline-yolov12s/train/val.log


# any-depth
## val on 1-gpu
mkdir -p ./runs/bdd100k/detect_wst/anydepth-yolov12l
python val_adn_bdd100k.py \
    --imgsz 1280 \
    --weight ./runs/bdd100k/detect/anydepth-yolov12s/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.0_orig/weights/best.pt \
    --project ./runs/bdd100k/detect/anydepth-yolov12s/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.0_orig \
    2>&1 | tee ./runs/bdd100k/detect/anydepth-yolov12s/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.0_orig/val_1280.log



python val_adn_bdd100k.py \
    --imgsz 640 \
    --weight ./runs/bdd100k/detect/anydepth-yolov12l/train_50e_SGD0900_bs64_nbs256_imgsz640_scale050/weights/best.pt \
    --project  ./runs/bdd100k/detect/anydepth-yolov12l/train_50e_SGD0900_bs64_nbs256_imgsz640_scale050 \
    2>&1 | tee ./runs/bdd100k/detect/anydepth-yolov12l/train_50e_SGD0900_bs64_nbs256_imgsz640_scale050/val.log



'''


