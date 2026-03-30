# ultralytics/models/yolo/mtl/val.py
from ultralytics.models.yolo.detect.val import DetectionValidator


class DetectionWSTValidator(DetectionValidator):
    """Validator for MTL model — detection metrics only for now."""
    pass