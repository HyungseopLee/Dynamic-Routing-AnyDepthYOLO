# @HyungseopLee
# ultralytics/models/yolo/mtl/train.py
from copy import copy
import torch
from ultralytics.data.dataset import DetectionWSTDataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionWSTModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import de_parallel


class DetectionWSTTrainer(DetectionTrainer):
    """Trainer for Multi-Task Learning: Detection + Attribute Classification."""

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return DetectionWSTDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionWSTModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1, data=self.data)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        from ultralytics.models.yolo.detect_wst.val import DetectionWSTValidator
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "weather_loss", "scene_loss", "timeofday_loss"
        return DetectionWSTValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        return batch