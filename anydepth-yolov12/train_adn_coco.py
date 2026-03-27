
# woochul: try to use local yolo, not from ultralytics 

#import sys
#import os
#sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import os 

from ultralytics import YOLO

# Load YOLOv10n model from scratch
# model = YOLO("yolo-ad-v12l.yaml")  # apply skippable in maximum
model = YOLO("yolo-ad-v12l.yaml")  # apply skippable in maximum
# model = YOLO("runs/exp8/train7/weights/last.pt")



# Train the model
results = model.train(
  optimizer='SGD', 
  momentum=0.937,  # default 0.937
  nbs=256, # default 256,

# experiment: the model seems to be underfitting with 0.0005 wd. 
  weight_decay=0.0005, # default 0.0005
  
  project='runs/exp8', 
  # project='runs/orig',
  data='coco.yaml',
  epochs=600,
  batch=64, #s:128, l:64, orig:256,
  imgsz=640,
  scale=0.9,  # n:0.5, S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.15,  # n:0.0; S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.5,  # n: 0.1; S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
  
  # Optimization settings for faster training
  workers=16,        # Increase data loading workers
  amp=True,         # Keep Automatic Mixed Precision (already default)
  #cache=True,       # Cache images in RAM for faster loading
  # compile=True,   # Disabled - doesn't help YOLO training speed

  # loss weights
  box = 7.5, # (float) box loss gain
  cls =0.5, # (float) cls loss gain (scale with pixels)
  dfl = 1.5, # (float) dfl loss gain

  # yolov10l_ad specific parameters
  kd_temp_dfl=1.0, #10.0, # (float) dfl kd temperature
  kd_temp_cls=2.0, #3.0, # 2.0, # (float) cls kd temperature
  kd_temp_feat=2.0, # 1.0,  
  # kd_temp_feat=2.0, # (float) feature kd temperature
  kd_cls= 0.4, #0.75, # 1.0, #0.5, # 0.5, # 0.75, # 0.75, # 0.5, # 1.5,  # 3.0, 
  kd_dfl= 0.8, # 1.0, # 1.5, # 1.5, # 0.75, # 1.5, # 3.0, 
  kd_box= 1.2, # 1.5, # 1.5, # 1.5, #, # 1.0, # 1.5, # 2.0, 
  kd_feat= 0.4, # 0.5, # 0.1, 
  alpha_base=0.2, # (float) loss weight for the base model
  kd_warmup_epochs=0, # (int) number of epochs to warmup kd losses
  
)
