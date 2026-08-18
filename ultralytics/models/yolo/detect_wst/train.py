# @HyungseopLee
# ultralytics/models/yolo/mtl/train.py

import math
from copy import copy

import numpy as np
import torch

from ultralytics.data.dataset import DetectionWSTDataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionWSTModel, DetectionWSTModelAnyDepth
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.torch_utils import (
    unwrap_model,
    autocast,
)

import time
import warnings
from ultralytics.utils import (
    colorstr,
)
from torch import distributed as dist

# @HyungseopLee
class DetectionWSTTrainer(DetectionTrainer):
    """Trainer for Multi-Task Learning: Detection + WST Classification."""

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



# @HyungseopLee
class DetectionWSTTrainerAnyDepth(DetectionTrainer):
    """
    Trainer for DetectionWSTModelAnyDepth.
    = DetectionTrainerAnyDepth (AnyDepth KD training loop)
    + DetectionWSTTrainer (WST dataset/model/validator)
    """
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
        from ultralytics.nn.tasks import DetectionWSTModelAnyDepth
        model = DetectionWSTModelAnyDepth(
            cfg, nc=self.data["nc"], verbose=verbose and RANK == -1, # data=self.data
        )
        if weights:
            model.load(weights)
        return model
    
    def get_validator(self):
        from ultralytics.models.yolo.detect_wst.val import DetectionWSTValidatorAnyDepth
        # super: box_loss, cls_loss, dfl_loss, weather_loss, scene_loss, timeofday_loss
        # base:  same
        # kd:    box_kd, cls_kd, dfl_kd, feat_kd
        self.loss_names = (
            "super/box", "super/cls", "super/dfl", "super/weather", "super/scene", "super/timeofday",
            "base/box",  "base/cls",  "base/dfl",  "base/weather",  "base/scene",  "base/timeofday",
            "kd/box",    "kd/cls",    "kd/dfl",    "kd/feat",
        )
        return DetectionWSTValidatorAnyDepth(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        
    def validate(self, skip=None):
        orig_loss_items = getattr(self, "loss_items", None)
        ref_tensor = getattr(self, "loss_items_super", getattr(self, "loss_items_base", None))
        if ref_tensor is not None:
            self.loss_items = torch.zeros_like(ref_tensor)
        else:
            self.loss_items = torch.zeros(6, device=self.device)
            
        metrics = self.validator(self, skip=skip)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy()) # use loss as fitness measure if not found
        
        # if not self.best_fitness or self.best_fitness < fitness:
        #     self.best_fitness = fitness
            
        # update best_fitness when only super model
        is_super = skip is None or not any(skip)
        if is_super and (not self.best_fitness or self.best_fitness < fitness):
            self.best_fitness = fitness
            
        self.loss_items = orig_loss_items
            
        return metrics, fitness
    
    
    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  
        t = time.time() - self.train_time_start
        row_vals = [self.epoch + 1, t] + vals
        row_str = ",".join([f"{v}" if isinstance(v, str) else f"{v:.6g}" for v in row_vals]) + "\n"
        
        with open(self.csv, "a") as f:
            f.write(s + row_str)
    
    
    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
 
        if RANK in {-1, 0}:
            print(self.model)
 
        nb = len(self.train_loader) # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1 # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
 
        epoch = self.start_epoch
        self.optimizer.zero_grad() # zero any resumed gradients to ensure stability on train start
 
        if RANK in {-1, 0}:
            try:
                from fvcore.nn import FlopCountAnalysis, flop_count_table
                class FLOPsWrapper(torch.nn.Module):
                    def __init__(self, base_model, skip):
                        super().__init__()
                        self.base_model = base_model
                        self.skip = skip
                    def forward(self, x):
                        return self.base_model(x, skip=self.skip)

                imgsz = self.args.imgsz
                if isinstance(imgsz, int):
                    imgsz = (imgsz, imgsz)
                num_skip = unwrap_model(self.model).num_skippable_layers
                dummy = torch.randn(1, 3, *imgsz).to(self.device)
                self.model.eval()

                # Super
                wrapper_super = FLOPsWrapper(unwrap_model(self.model), [False] * num_skip)
                flops_super = FlopCountAnalysis(wrapper_super, dummy)
                LOGGER.info(f"\n[FLOPs] Super model (skip=False) @ {imgsz[0]}x{imgsz[1]}:")
                LOGGER.info(flop_count_table(flops_super))
                LOGGER.info(f"  MACs  (super): {flops_super.total() / 1e9:.2f} GMACs")
                LOGGER.info(f"  FLOPs (super): {flops_super.total() * 2 / 1e9:.2f} GFLOPs")

                # Base
                wrapper_base = FLOPsWrapper(unwrap_model(self.model), [True] * num_skip)
                flops_base = FlopCountAnalysis(wrapper_base, dummy)
                LOGGER.info(f"\n[FLOPs] Base model (skip=True) @ {imgsz[0]}x{imgsz[1]}:")
                LOGGER.info(flop_count_table(flops_base))
                LOGGER.info(f"  MACs  (base): {flops_base.total() / 1e9:.2f} GMACs")
                LOGGER.info(f"  FLOPs (base): {flops_base.total() * 2 / 1e9:.2f} GFLOPs")

            except Exception as e:
                LOGGER.warning(f"[Warning] Failed to calculate FLOPs: {e}")
 
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
 
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
 
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
 
            self.tloss_super = None
            self.tloss_base  = None
            self.tloss_kd    = None
 
            skip = [False,] * unwrap_model(self.model).num_skippable_layers
 
            for i, batch in enumerate(self.train_loader):
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw] # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    
                    # ── Forward super (skip=False) ──────────────────────────
                    skip_super = [False,] * len(skip)
                    preds_super = self.model(batch["img"], skip=skip_super, return_features=True)
                    
                    
                    # # @HyungseopLee: preds_super
                    # print(f"\n=== [DEBUG] preds_super ===")
                    # print(f"preds_super.keys(): {preds_super.keys()}")
                    # # 1. pred
                    # print("1. 'pred' (List of Tensors):")
                    # for i, p in enumerate(preds_super.get('pred', [])):
                    #     print(f"   [{i}] shape: {p.shape}")
                    # # 2. features
                    # print("2. 'features' (List of Tensors):")
                    # for i, f in enumerate(preds_super.get('features', [])):
                    #     print(f"   [{i}] shape: {f.shape}")
                    # # 3. attr_out
                    # print("3. 'attr_out' (Dict of Tensors):")
                    # attr_out = preds_super.get('attr_out', {})
                    # if isinstance(attr_out, dict):
                    #     for k, v in attr_out.items():
                    #         print(f"   ['{k}'] shape: {v.shape}")
                    # else:
                    #     print(f"   attr_out is not dict, it's {type(attr_out)}")
                    # print("================================\n")
                    '''
                    preds_super (dict):
                        'pred': [ # detection out
                            tensor[b, 74, 80, 80], # P3: low-level
                            tensor[b, 74, 40, 40], # P4: mid-level
                            tensor[b, 74, 20, 20]  # P5: high-level
                        ],
                        'features': [ # intermediate features
                            tensor[b, c], # 0 (intermediate features for KD)
                            ...           # 1 ~ 5
                            tensor[b, c]  # 6
                        ],
                        'attr_out': { # attributes(WST) out
                            'weather':   tensor[b, 6],
                            'scene':     tensor[b, 6],
                            'timeofday': tensor[b, 3]
                        }
                    '''

                    # @HyungseopLee: give all raw preds_super (det_out, features, attr_out) to loss() for training Detect+WST, below code commented out (preds_) is det_out only.
                    # # Extract predictions from dictionary if return_features=True
                    # if isinstance(preds_super, dict) and "pred" in preds_super:
                    #     preds_ = preds_super["pred"]
                    # else:
                    #     preds_ = preds_super
                    loss_super, self.loss_items_super = unwrap_model(self.model).loss(batch, preds_super) # preds_ is only det_out

                    self.loss_super = loss_super.sum()
                    if RANK != -1:
                        self.loss_super *= world_size
                    self.tloss_super = (
                        (self.tloss_super * i + self.loss_items_super) / (i + 1) if self.tloss_super is not None else self.loss_items_super
                    )
                    
                    # Backward super
                    with torch.amp.autocast('cuda', enabled=False):
                        # print("XXXXXX Backward super XXXXXX")
                        self.scaler.scale(self.loss_super).backward()
 
                    # ── Forward base (skip=True) ────────────────────────────
                    skip_base = [True] * len(skip)
                    preds_base = self.model(batch["img"], skip=skip_base, return_features=True)
 
                    # MTL(Detect + WST) loss + KD loss for base model
                    loss_base, self.loss_items_base = unwrap_model(self.model).loss(batch, preds_base)
                    self.loss_base = loss_base.sum()
                    if RANK != -1:
                        self.loss_base *= world_size
                    self.tloss_base = (
                        (self.tloss_base * i + self.loss_items_base) / (i + 1) if self.tloss_base is not None else self.loss_items_base
                    )
 
                    # ── KD loss (super -> base) ──────────────────────────────
                    loss_kd, self.loss_items_kd = unwrap_model(self.model).loss(
                        batch, 
                        preds=preds_super, 
                        preds_base=preds_base
                    )
                    self.loss_kd = loss_kd.sum()
                    
                    if RANK != -1:
                        self.loss_kd *= world_size
                    self.tloss_kd = (
                        (self.tloss_kd * i + self.loss_items_kd) / (i + 1)
                        if self.tloss_kd is not None else self.loss_items_kd
                    )
 
                    # ── Total loss = alpha * base + (1-alpha) * kd ──────────
                    kd_scale = 1.0
                    self.loss = self.args.alpha_base * self.loss_base + \
                        (1 - self.args.alpha_base) * self.loss_kd * kd_scale
        
                    # Backward base
                    with torch.amp.autocast('cuda', enabled=False):
                        # print("XXXXXX Backward base XXXXXX")
                        self.scaler.scale(self.loss).backward()
                        
 
                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    
                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0) # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop: # training time exceeded
                            break
 
                # Log
                if RANK in {-1, 0}:
                    if i % max(1, nb // 10) == 0:
                        # WST items: box, cls, dfl, weather, scene, timeofday
                        si, bi, ki = self.loss_items_super, self.loss_items_base, self.loss_items_kd
                        progress_msg = (
                            f"Epoch {epoch+1}/{self.epochs}, Batch {i+1}/{nb} ({100*(i+1)/nb:.1f}%) | "
                            f"super box,cls,dfl,W,S,T: {si[0]:.3f},{si[1]:.3f},{si[2]:.3f},{si[3]:.3f},{si[4]:.3f},{si[5]:.3f} | "
                            f"base  box,cls,dfl,W,S,T: {bi[0]:.3f},{bi[1]:.3f},{bi[2]:.3f},{bi[3]:.3f},{bi[4]:.3f},{bi[5]:.3f} | "
                            f"kd box,cls,dfl,feat: {ki[0]:.5f},{ki[1]:.5f},{ki[2]:.5f},{ki[3]:.5f}"
                        )
                        LOGGER.info(progress_msg)
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
 
                self.run_callbacks("on_train_batch_end")
                break
                
 
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
 
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
 
                train_loss_keys = ["train/box", "train/cls", "train/dfl", "train/weather", "train/scene", "train/timeofday"]
 
                # Validation: super
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate(skip=[False] * len(skip))
                clean_metrics_super = {k.replace("super/", "").replace("base/", ""): v for k, v in self.metrics.items()}
                train_loss_super = dict(zip(train_loss_keys, self.tloss_super))
                self.save_metrics(metrics={
                    "config": "super",
                    **train_loss_super,
                    **clean_metrics_super,
                    **self.lr
                })
 
                # Validation: base
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, _ = self.validate(skip=[True] * len(skip))
                clean_metrics_base = {k.replace("super/", "").replace("base/", ""): v for k, v in self.metrics.items()}
                train_loss_base = dict(zip(train_loss_keys, self.tloss_base))
                self.save_metrics(metrics={
                    "config": "base",
                    **train_loss_base,
                    **clean_metrics_base,
                    **self.lr
                })
 
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
 
                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
 
            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch # do not move
                self.stop |= epoch >= self.epochs # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()
 
            # Early Stopping
            if RANK != -1: # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0) # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break # must break all DDP ranks
            epoch += 1
 
        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")