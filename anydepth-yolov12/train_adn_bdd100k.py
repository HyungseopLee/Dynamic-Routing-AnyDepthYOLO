from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', type=int, default=640, help='Validation image size')
parser.add_argument('--project', type=str, default='')
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--data', type=str, default='')
args = parser.parse_args()


model = YOLO(args.weight, task='detect_wst')


# Train the model
results = model.train(
  task='detect_wst', # ('w'eather, 's'cene, 't'imeofday)
  optimizer='SGD', 
  momentum=0.937,  # default 0.937
  nbs=256, # default 256,

  project=args.project,
  data=args.data,
  epochs=600,
  batch=64, #s:128, l:64, orig:256,
  imgsz=args.imgsz,
  scale=0.9,  # n:0.5, S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.15,  # n:0.0; S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.5,  # n: 0.1; S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1",
)

'''
# COCO pretrained weight path
baseline: 
  ./pretrained/yolov12l.pt
any-depth:
  ./pretrained/yolo-ad-exp8_105_epoch539_0.539_0.520.pt



# 2 GPU
mkdir -p ./runs/bdd100k/baseline-yolov12l
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node 2 train_adn_bdd100k.py \
  --data bdd100k.yaml \
  --imgsz 640 \
  --weight ./pretrained/yolov12l.pt \
  --project ./runs/bdd100k/baseline-yolov12l \
  2>&1 | tee ./runs/bdd100k/baseline-yolov12l/test.log


'''

