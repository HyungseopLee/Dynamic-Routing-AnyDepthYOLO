
# woochul: try to use local yolo, not from ultralytics 

#import sys
#import os
#sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import os 

from ultralytics import YOLO

# Load YOLOv12 AnyDepth model - first initialize with YAML, then load weights
# model = YOLO("yolo-ad-v12l.yaml")  # Initialize with AnyDepth architecture
model = YOLO("yolo-ad-v12n.yaml")  # Initialize with AnyDepth architecture
#model = YOLO("runs/exp4/train2/weights/last.pt")  # Load pretrained weights

# Train the model
results = model.train(
  # default hyperparameters for yolov12 in neurips paper
  # but, auto optimizer set SGD momentum=0.9 with bs 64, ns 64 makes good results
  # so, I will disable it for a while
  optimizer='SGD', 
  momentum=0.937,
  nbs=256,
  
  project='runs/exp5', 
  data='coco.yaml',
  epochs=600,
  batch=256, #s:256, l:64, orig:256,
  imgsz=640,
  scale=0.5,  # n:0.5, S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # n:0.0; S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # n: 0.1; S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1",
  
  # Optimization settings for faster training
  workers=8,        # Increase data loading workers
  amp=True,         # Keep Automatic Mixed Precision (already default)
  # cache=True,       # Cache images in RAM for faster loading

  # loss weights
  box = 7.5, # (float) box loss gain
  cls =0.5, # (float) cls loss gain (scale with pixels)
  dfl = 1.5, # (float) dfl loss gain

  # yolov10l_ad specific parameters
  kd_temp=1.0, 
  kd_cls= 0.75, # 1.5,  # 3.0, 
  kd_dfl= 1.5, # 3.0, 
  kd_box= 1.0,  # 1.5, # 2.0, 
  kd_feat=0.5, 
  
#  resume=True,
)
