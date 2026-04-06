# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.nn.tasks import DetectionModelAnyDepth  # woochul: YOLOv12 AnyDepth
from ultralytics.nn.tasks import DetectionWSTModel, DetectionWSTModelAnyDepth # @HyungseopLee
from ultralytics.utils import ROOT, yaml_load
from ultralytics.utils import RANK


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        
        # @HyungseopLee: debugging
        if RANK in {-1, 0}:
            print(f"task: {task}")
            
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yolo-ad" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:    # woochul: YOLOv12 AnyDepth
            print("Loading YOLOv12 Any-Depth model from", path)
            # new_instance = YOLOv12AnyDepth(path)
            new_instance = YOLOv12AnyDepth(path, task=task, verbose=verbose) # @HyungseopLee: 
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)
            
            # After loading, check if it's an AnyDepth model and switch if needed
            if self._is_loaded_anydepth_model():
                print("Detected YOLOv12AnyDepth model from weights, switching to AnyDepth task_map")
                self._switch_to_anydepth_mode()
    
    def _is_loaded_anydepth_model(self):
        """Check if loaded weights are from YOLOv12AnyDepth model."""            

        # Check if model has skippable/switchable layers
        if hasattr(self.model, 'model'):
            for module in self.model.model.modules():
                if any(name in module.__class__.__name__ for name in ['Skippable', 'SwitchableConv']):
                    return True
        
        return False
    
    def _switch_to_anydepth_mode(self):
        """Switch to YOLOv12AnyDepth task_map behavior."""
        # Set flag to use AnyDepth version of task_map
        self._use_anydepth_task_map = True

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        _map = {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
            # @HyungseopLee
            "detect_wst": { 
                "model": DetectionWSTModel,
                "trainer": yolo.detect_wst.DetectionWSTTrainer,
                "validator": yolo.detect_wst.DetectionWSTValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }

        # Check if we should use AnyDepth task_map
        if (hasattr(self, '_use_anydepth_task_map') and self._use_anydepth_task_map):
            _map['detect'] = {
                "model": DetectionModelAnyDepth,
                "trainer": yolo.detect.DetectionTrainerAnyDepth,
                "validator": yolo.detect.DetectionValidatorAnyDepth,
                "predictor": yolo.detect.DetectionPredictor,
            }
            _map['detect_wst'] = { # @HyungseopLee: Anydepth mode + task:'detect_wst'
                "model": DetectionWSTModelAnyDepth,
                "trainer": yolo.detect_wst.DetectionWSTTrainerAnyDepth,
                "validator": yolo.detect_wst.DetectionWSTValidatorAnyDepth,
                "predictor": yolo.detect.DetectionPredictor,
            }
        return _map


class YOLOv12AnyDepth(Model):
    def __init__(self, model: str | Path = "yolo12ad.pt", task=None, verbose: bool = False) -> None:
        # super().__init__(model=model, task="detect", verbose=verbose)
        super().__init__(model=model, task=task or "detect", verbose=verbose) # @HyungseopLee: no hardcoding

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModelAnyDepth,
                "trainer": yolo.detect.DetectionTrainerAnyDepth,
                "validator": yolo.detect.DetectionValidatorAnyDepth,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "detect_wst": {
                "model": DetectionWSTModelAnyDepth,
                "trainer": yolo.detect_wst.DetectionWSTTrainerAnyDepth,
                "validator": yolo.detect_wst.DetectionWSTValidatorAnyDepth,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }



class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
