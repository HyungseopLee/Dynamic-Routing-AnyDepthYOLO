from ultralytics import YOLO 
from ultralytics.utils.torch_utils import model_info
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', type=int, default=640, help='Validation image size')
parser.add_argument('--project', type=str, default='')
parser.add_argument('--weight', type=str, default='')
args = parser.parse_args()
IMG_SIZE = args.imgsz
print(f"[*] Running validation with Image Size: {IMG_SIZE}")

model = YOLO(args.weight)


print(f"model: {model}")

model.val(
    data='coco.yaml', 
    save_json=True, 
    project=args.project,
    name=f'Full_{IMG_SIZE}',
    imgsz=IMG_SIZE,
)


'''

# weight path
baseline: ./pretrained/yolov12l.pt



# val on 1-gpu
mkdir -p runs/baseline-yolov12l
python val_baseline.py \
    --imgsz 640 \
    --weight ./pretrained/yolov12l.pt \
    --project runs/baseline-yolov12l \
    2>&1 | tee ./runs/baseline-yolov12l/val_640.log


'''


