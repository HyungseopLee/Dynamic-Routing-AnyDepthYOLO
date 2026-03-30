# @HyungseopLee: ultralytics/models/yolo/detect_wst/val.py

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils import RANK


class DetectionWSTValidator(DetectionValidator):
    """
    Validator for DetectionWSTModel.
    Detection metrics (mAP) + Attribute classification accuracy (weather, scene, timeofday)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_correct = {"weather": 0, "scene": 0, "timeofday": 0}
        self.attr_total   = {"weather": 0, "scene": 0, "timeofday": 0}
        self.attr_preds   = []  # save for (attr_out, batch)

    def postprocess(self, preds):
        """Extract det_out from (det_out, attr_out) tuple."""
        if isinstance(preds, tuple):
            self._attr_out = preds[1]  # save attr_out
            preds = preds[0]
        else:
            self._attr_out = None
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        """Detection metrics + attribute accuracy."""
        # detection metrics
        super().update_metrics(preds, batch)

        # attribute accuracy
        if self._attr_out is not None:
            for key in ("weather", "scene", "timeofday"):
                gt = batch[key].to(self.device)          # (B,)
                pred = self._attr_out[key].argmax(dim=1) # (B,)

                # ignore_index=-1
                valid = gt != -1
                if valid.sum() > 0:
                    self.attr_correct[key] += (pred[valid] == gt[valid]).sum().item()
                    self.attr_total[key]   += valid.sum().item()

    def get_stats(self):
        """Detection stats + attribute accuracy."""
        stats = super().get_stats()

        # get attribute accuracy
        for key in ("weather", "scene", "timeofday"):
            total = self.attr_total[key]
            acc = self.attr_correct[key] / total if total > 0 else 0.0
            stats[f"metrics/acc_{key}"] = acc

        return stats

    def get_desc(self):
        """Return description with attribute accuracy columns."""
        return ("%22s" + "%11s" * 6 + "%11s" * 3) % (
            "Class", "Images", "Instances",
            "Box(P", "R", "mAP50", "mAP50-95)",
            "Acc_W", "Acc_S", "Acc_T",
        )

    def print_results(self):
        """Print detection + attribute results."""
        super().print_results()
        if RANK in {-1, 0}:
            print("\nAttribute Classification Accuracy:")
            print(f"  {'Attribute':<12} {'Acc':>8} {'Correct':>10} {'Total':>10}")
            print(f"  {'-'*42}")
            for key in ("weather", "scene", "timeofday"):
                total = self.attr_total[key]
                acc = self.attr_correct[key] / total if total > 0 else 0.0
                print(f"  {key:<12} {acc:>8.4f} {self.attr_correct[key]:>10} {total:>10}")


    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics()
        # add attr accuracy to metrics.results_dict
        for key in ("weather", "scene", "timeofday"):
            total = self.attr_total[key]
            acc = self.attr_correct[key] / total if total > 0 else 0.0
            self.metrics.results_dict[f"metrics/acc_{key}"] = acc