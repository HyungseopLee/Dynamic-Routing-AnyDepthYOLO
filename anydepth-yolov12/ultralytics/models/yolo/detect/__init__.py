# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

from .train import DetectionTrainerAnyDepth  # woochul: YOLOv12 AnyDepth
from .val import DetectionValidatorAnyDepth  # woochul: YOLOv12 AnyDepth

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "DetectionTrainerAnyDepth", "DetectionValidatorAnyDepth"
