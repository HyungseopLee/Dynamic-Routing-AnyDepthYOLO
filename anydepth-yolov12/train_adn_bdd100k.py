from ultralytics import YOLO
import argparse
from ultralytics.utils import RANK

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='')
parser.add_argument('--epoch', type=int, default=50, help='Epoch')
parser.add_argument('--imgsz', type=int, default=640, help='Image size (long side, e.g., 640 or 1280)')
parser.add_argument('--project', type=str, default='')
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--task', type=str, default='')
parser.add_argument('--data', type=str, default='')
parser.add_argument('--resume', action='store_true', help='Resume training from the provided weight')

args = parser.parse_args()
IMG_SIZE = args.imgsz

if RANK in {-1, 0}:
  print(f"[*] Image size: {IMG_SIZE}")


if args.resume:
  
  if RANK in {-1, 0}:
      print(f"[*] Resuming training from {args.weight}")
  model = YOLO(args.weight, task=args.task)
  results = model.train(resume=True, rect=True)
    
else:
    if RANK in {-1, 0}:
      print(f"[*] Starting new training using weight: {args.weight}")
    
    model = YOLO(args.config, task=args.task)
    model.load(args.weight)

    # Train the model
    results = model.train(
      task=args.task,
      project=args.project,
      data=args.data,
      
      # epoch, optimizer, batch size, lr
      epochs=args.epoch,
      optimizer='SGD', 
      momentum=0.900,  # default 0.937
      batch=32, #s:128, l:64, orig:256,
      nbs=256, # default 256,
      lr0=1e-3, # initial lr: fromscratch=1e-2, finetuning:1e-3 or 1e-4
      lrf=1e-2, # lr0 ~ (lr0 * lrf)
      
      # image
      imgsz=IMG_SIZE, 
      rect=True, # for 1280x720 (bdd100k). not 1280x1280
      
      # # single-? multi-scale?
      # scale=0.9,  # (float) image scale (+/- gain) (n:0.5, S:0.9; M:0.9; L:0.9; X:0.9)
      
      # data augmentation
      mosaic=0.0, # mosaic augmentation prob.
      close_mosaic=0, # disable mosaic augmentation for final epochs (0 to disable)
      mixup=0.0,       # n:0.0; S:0.05; M:0.15; L:0.15; X:0.2
      copy_paste=0.0,  # n:0.1; S:0.15; M:0.4; L:0.5; X:0.6
      flipud=0.0, # (float) image flip up-down (probability)rect=
      
      
      device="0,1,2,3",
    )

'''

# config
  baseline: 
    ./ultralytics/cfg/models/v12/yolov12l.yaml
  any-depth:
    ./ultralytics/cfg/models/v12/yolo-ad-v12l.yaml

# weight
  baseline: 
    ./pretrained/yolov12l.pt
  any-depth:
    ./pretrained/yolo-ad-exp8_105_epoch539_0.539_0.520.pt



# Baseline
## 2 GPU
mkdir -p ./runs/bdd100k/detect/baseline-yolov12l
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.run --nproc_per_node 4 train_adn_bdd100k.py \
  --task detect \
  --config ./ultralytics/cfg/models/v12/yolov12l.yaml \
  --data bdd100k.yaml \
  --epoch 30 \
  --imgsz 1280 \
  --weight ./pretrained/yolov12l.pt \
  --project ./runs/bdd100k/detect/baseline-yolov12l \
  2>&1 | tee ./runs/bdd100k/detect/baseline-yolov12l/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-740_singleScale_augNothing.log


# Any-depth
## 2 GPU
mkdir -p ./runs/bdd100k/detect/anydepth-yolov12l
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node 2 train_adn_bdd100k.py \
  --task detect \
  --config ./ultralytics/cfg/models/v12/yolo-ad-v12l.yaml \
  --data bdd100k.yaml \
  --epoch 30 \
  --imgsz 1280 \
  --weight ./pretrained/yolo-ad-exp8_105_epoch539_0.539_0.520.pt \
  --project ./runs/bdd100k/detect/anydepth-yolov12l \
  2>&1 | tee ./runs/bdd100k/detect/anydepth-yolov12l/100e_SGD0900_bs32_nbs256_1e-3_1e-5_640-360_singleScale_augNothing_volta.log


'''

