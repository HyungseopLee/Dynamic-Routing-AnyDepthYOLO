import torch
from ultralytics import YOLO 
from ultralytics.utils.torch_utils import model_info
# Load the YOLO model with specific scale
# model = YOLO('yolo11l.yaml')

model = YOLO('pretrain/yolo-ad-exp8_105_epoch539_0.539_0.520.pt')
# model = YOLO('yolo-ad-v12l.yaml')


# skip = [True,] * model.num_skippable_layers
skip = [False,] * model.num_skippable_layers

# skip only in the neck
# skip = [False, False, False, False, True, True, True, True]

# skip only in the backbone
# skip = [True, True, True, True, False, False, False, False]


model.val(data='coco.yaml', save_json=True, skip=skip)

#x = torch.randn(1, 3, 640, 640)
#skip = [False,] * model.num_skippable_layers
#print(f"------skip: {skip}----")
#pred = model(x, skip=skip)  # forward pass
# print(pred)



