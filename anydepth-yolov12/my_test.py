import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
# Load the YOLO model with specific scale
# model = YOLO('yolo11l.yaml')
model = YOLO('yolo-ad-v12l.yaml')  # n/s/l 
#model = YOLO('yolov12n.yaml')  # n/s/m/l/x

# Access the underlying PyTorch model
# pytorch_model = model.model

print(model.model)

model_info(model.model, verbose=True)

#x = torch.randn(1, 3, 640, 640)
#skip = [False,] * model.num_skippable_layers
#print(f"------skip: {skip}----")
#pred = model(x, skip=skip)  # forward pass
# print(pred)



