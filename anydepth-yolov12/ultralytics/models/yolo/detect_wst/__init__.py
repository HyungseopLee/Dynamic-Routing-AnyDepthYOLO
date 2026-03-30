# ultralytics/models/yolo/detect_wst/__init__.py
from .train import DetectionWSTTrainer
from .val import DetectionWSTValidator

__all__ = "DetectionWSTTrainer", "DetectionWSTValidator"