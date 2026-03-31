# @HyungseopLee: ultralytics/models/yolo/detect_wst/val.py

from pathlib import Path
import numpy as np

import torch

from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import RANK

from ultralytics.utils.torch_utils import smart_inference_mode
from fvcore.nn import FlopCountAnalysis, flop_count_table


class DetectionWSTValidator(DetectionValidator):
    """
    Validator for DetectionWSTModel.
    Detection metrics (mAP) + Attribute classification accuracy (weather, scene, timeofday)
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "detect_wst"
        self.attr_correct = {"weather": 0, "scene": 0, "timeofday": 0}
        self.attr_total   = {"weather": 0, "scene": 0, "timeofday": 0}
        self._attr_out    = None
        self.attr_preds_all = {"weather": [], "scene": [], "timeofday": []}
        self.attr_gts_all   = {"weather": [], "scene": [], "timeofday": []}
    
    def init_metrics(self, model):
        """Initialize metrics including attribute classification metrics."""
        super().init_metrics(model)
        for key in ("acc_weather", "acc_scene", "acc_timeofday"):
            self.metrics.results_dict[f"metrics/{key}"] = 0.0
        self.attr_correct   = {"weather": 0, "scene": 0, "timeofday": 0}
        self.attr_total     = {"weather": 0, "scene": 0, "timeofday": 0}
        self.attr_undefined = {"weather": 0, "scene": 0, "timeofday": 0}
        self.attr_preds_all = {"weather": [], "scene": [], "timeofday": []}
        self.attr_gts_all   = {"weather": [], "scene": [], "timeofday": []}
        self._attr_out      = None

    def get_stats(self):
        stats = super().get_stats()
        for key in ("weather", "scene", "timeofday"):
            total = self.attr_total[key]
            acc   = self.attr_correct[key] / total if total > 0 else 0.0
            stats[f"metrics/acc_{key}"] = acc
        return stats

    def preprocess(self, batch):
        """Preprocess batch including attribute labels."""
        batch = super().preprocess(batch)
        for key in ("weather", "scene", "timeofday"):
            if key in batch:
                batch[key] = batch[key].to(self.device)
        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build DetectionWSTDataset."""
        from ultralytics.data.dataset import DetectionWSTDataset
        return DetectionWSTDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=self.args.rect,
            cache=False,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader using DetectionWSTDataset."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers * 2, shuffle=False, rank=-1)

    def postprocess(self, preds):
        self._attr_out = None
        if isinstance(preds, list) and len(preds) == 2 and isinstance(preds[1], dict):
            # [det_out, attr_out]
            self._attr_out = preds[1]
            preds = preds[0]  # det_out (tuple)
        elif isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict):
            # (det_out, attr_out)
            self._attr_out = preds[1]
            preds = preds[0]  # det_out
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        if self._attr_out is not None:
            for key in ("weather", "scene", "timeofday"):
                if key not in batch:
                    continue
                gt   = batch[key].to(self.device)
                pred = self._attr_out[key].argmax(dim=1)
                valid = gt != -1
                if valid.sum() > 0:
                    self.attr_correct[key] += (pred[valid] == gt[valid]).sum().item()
                    self.attr_total[key]   += valid.sum().item()
                    self.attr_preds_all[key].extend(pred[valid].cpu().tolist())
                    self.attr_gts_all[key].extend(gt[valid].cpu().tolist())


    def print_results(self):
        if RANK not in {-1, 0}:
            return
        
        # Detection result
        print(self.get_desc())
        super().print_results()

        # WST result
        WEATHER_NAMES   = ["clear", "rainy", "snowy", "overcast", "foggy", "partly cloudy"]
        SCENE_NAMES     = ["city street", "highway", "residential", "parking lot", "tunnel", "gas stations"]
        TIMEOFDAY_NAMES = ["daytime", "night", "dawn/dusk"]
        NAMES_MAP = {
            "weather":   WEATHER_NAMES,
            "scene":     SCENE_NAMES,
            "timeofday": TIMEOFDAY_NAMES,
        }


        for key in ("weather", "scene", "timeofday"):
            gts   = self.attr_gts_all[key]
            preds = self.attr_preds_all[key]
            if len(gts) == 0:
                continue

            names   = NAMES_MAP[key]
            nc      = len(names)
            total   = self.attr_total[key]
            acc     = self.attr_correct[key] / total * 100 if total > 0 else 0.0

            # per-class TP, FP, FN 계산
            tp = [0] * nc
            fp = [0] * nc
            fn = [0] * nc

            for g, p in zip(gts, preds):
                if g == p:
                    tp[g] += 1
                else:
                    fp[p] += 1
                    fn[g] += 1

            print(f"\n{'='*70}")
            print(f"  [{key.upper()}]  Overall Acc: {acc:.2f}%  ({self.attr_correct[key]}/{total})")
            print(f"{'='*70}")
            print(f"  {'Class':<20} {'Correct/Total':>15} {'Acc':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
            print(f"  {'-'*70}")

            for i, name in enumerate(names):
                support  = tp[i] + fn[i]
                correct  = tp[i]
                cls_acc  = correct / support * 100 if support > 0 else 0.0
                prec     = tp[i] / (tp[i] + fp[i]) * 100 if (tp[i] + fp[i]) > 0 else 0.0
                rec      = tp[i] / (tp[i] + fn[i]) * 100 if (tp[i] + fn[i]) > 0 else 0.0
                f1       = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                print(f"  {name:<20} {correct:>6}/{support:<8} {cls_acc:>7.2f}% {prec:>9.2f}% {rec:>7.2f}% {f1:>7.2f}%")

            print(f"  {'-'*70}")
            print(f"  {'Overall':<20} {self.attr_correct[key]:>6}/{total:<8} {acc:>7.2f}%")
           


class DetectionWSTValidatorAnyDepth(DetectionWSTValidator):
    """
    Validator for DetectionWSTModelAnyDepth.
    """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.save_json |= self.is_coco

    @smart_inference_mode()    
    def __call__(self, trainer=None, model=None, skip=None):
        if skip is not None:
            # print(f"Using skip={skip} for DetectionWSTValidatorAnyDepth")
            self.skip = skip
        
        model_args = {'skip': self.skip} if hasattr(self, 'skip') else {}

        target_model = model or getattr(trainer, 'model', None) or getattr(self, 'model', None)
        
        if target_model is not None and FlopCountAnalysis is not None:
            class FLOPsWrapper(torch.nn.Module):
                def __init__(self, base_model, skip_args):
                    super().__init__()
                    self.base_model = base_model
                    self.skip_args = skip_args
                def forward(self, x):
                    # 모델 포워드 시 skip 인자를 함께 전달
                    return self.base_model(x, **self.skip_args)
            
            imgsz = self.args.imgsz
            if isinstance(imgsz, int):
                imgsz = (imgsz, imgsz)
            device = next(target_model.parameters()).device
            dummy_input = torch.randn(1, 3, *imgsz).to(device)
            
            try:
                wrapper = FLOPsWrapper(target_model, model_args)
                flops = FlopCountAnalysis(wrapper, dummy_input)
                print(f"\n[*] FLOPs calculated at resolution: {imgsz[0]}x{imgsz[1]}")
                print(flop_count_table(flops))
            except Exception as e:
                print(f"[Warning] Failed to calculate FLOPs: {e}")

        # 3. DetectionValidator의 핵심 검증 루프 실행
        # super(DetectionWSTValidator, self).__call__ 을 호출함으로써 
        # Base 클래스의 검증 루프를 돌면서 DetectionWSTValidator에 정의된 
        # preprocess, postprocess, update_metrics 등이 자연스럽게 실행됩니다.
        return super(DetectionWSTValidator, self).__call__(trainer=trainer, model=model, model_args=model_args)