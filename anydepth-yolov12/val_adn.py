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

# skip only in the neck
# skip = [False, False, False, False, True, True, True, True]

# skip only in the backbone
# skip = [True, True, True, True, False, False, False, False]


# Full
skip = [False,] * model.num_skippable_layers
model.val(
    data='coco.yaml', 
    save_json=True, 
    skip=skip,
    project=args.project,
    name=f'Full_{IMG_SIZE}',
    imgsz=IMG_SIZE,
)

# Base
skip = [True,] * model.num_skippable_layers
model.val(
    data='coco.yaml', 
    save_json=True, 
    skip=skip,
    project=args.project,
    name=f'Base_{IMG_SIZE}',
    imgsz=IMG_SIZE,
)


'''

# weight path
(coco) anydepth: ./pretrained/yolo-ad-exp8_105_epoch539_0.539_0.520.pt



# val on 1-gpu
mkdir -p runs/coco/anydepth-yolov12l
python val_adn.py \
    --imgsz 640 \
    --weight ./pretrained/yolo-ad-exp8_105_epoch539_0.539_0.520.pt \
    --project runs/coco/anydepth-yolov12l \
    2>&1 | tee ./runs/coco/anydepth-yolov12l/val_640.log

'''


