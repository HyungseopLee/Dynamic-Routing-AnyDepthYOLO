from ultralytics import YOLO 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', type=int, default=640, help='Validation image size')
parser.add_argument('--project', type=str, default='')
parser.add_argument('--weight', type=str, default='')
args = parser.parse_args()
IMG_SIZE = args.imgsz
print(f"[*] Running validation with Image Size: {IMG_SIZE}")

model = YOLO(args.weight, task='detect_wst')


print(f"model: {model}")


# Full
model.val(
    task='detect_wst',
    data='bdd100k.yaml', 
    save_json=True, 
    # skip=skip,
    project=args.project,
    name=f'Full_{IMG_SIZE}',
    imgsz=IMG_SIZE,
)

# # Base
# skip = [True,] * model.num_skippable_layers
# model.val(
#     data='coco.yaml', 
#     save_json=True, 
#     skip=skip,
#     project=args.project,
#     name=f'Base_{IMG_SIZE}',
#     imgsz=IMG_SIZE,
# )


'''

# weight path
(bdd100k) baseline: ./runs/bdd100k/baseline-yolov12l/train5/weights/best.pt



# val on 1-gpu
mkdir -p runs/bdd100k/baseline-yolov12l
python val_adn_bdd100k.py \
    --imgsz 640 \
    --weight ./runs/bdd100k/baseline-yolov12l/train5/weights/best.pt \
    --project runs/bdd100k/baseline-yolov12l \
    2>&1 | tee ./runs/bdd100k/baseline-yolov12l/val_640.log

'''


