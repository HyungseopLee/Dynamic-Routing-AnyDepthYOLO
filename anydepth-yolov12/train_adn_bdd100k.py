from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', type=int, default=640, help='Validation image size')
parser.add_argument('--project', type=str, default='')
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--task', type=str, default='')
parser.add_argument('--data', type=str, default='')
args = parser.parse_args()


model = YOLO(args.weight, task=args.task)


# Train the model
results = model.train(
  task=args.task, # ('w'eather, 's'cene, 't'imeofday)
  optimizer='SGD', 
  momentum=0.900,  # default 0.937
  nbs=256, # default 256,

  project=args.project,
  data=args.data,
  epochs=50,
  batch=64, #s:128, l:64, orig:256,
  imgsz=args.imgsz,
  
  wst=1.0, # loss weighting for (weather loss + scene loss + time loss)
  
  scale=0.5,  # n:0.5, S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=0.0,
  close_mosaic=0, # disable mosaic augmentation for final epochs (0 to disable)
  mixup=0.0,  # @HyungseopLee: no need for finetuning
  copy_paste=0.0,  # @HyungseopLee: no need for finetuning
  device="0,1",
)

'''
# COCO pretrained weight path
baseline: 
  ./pretrained/yolov12l.pt
any-depth:
  ./pretrained/yolo-ad-exp8_105_epoch539_0.539_0.520.pt


# task: "detect", "detect_wst"


# 2 GPU
mkdir -p ./runs/bdd100k/detect_wst/baseline-yolov12l
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node 2 train_adn_bdd100k.py \
  --task detect_wst \
  --data bdd100k.yaml \
  --imgsz 640 \
  --weight ./pretrained/yolov12l.pt \
  --project ./runs/bdd100k/detect_wst/baseline-yolov12l \
  2>&1 | tee ./runs/bdd100k/detect_wst/baseline-yolov12l/train_50e_SGD0900_bs64_nbs256_imgsz640_scale050_wst1.0.log


'''

