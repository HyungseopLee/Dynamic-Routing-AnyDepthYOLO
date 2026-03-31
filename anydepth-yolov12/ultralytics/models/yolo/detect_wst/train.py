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
            cfg, nc=self.data["nc"], verbose=verbose and RANK == -1, data=self.data
        )
        if weights:
            model.load(weights)
        return model
    
    def get_validator(self):
        from ultralytics.models.yolo.detect_wst.val import DetectWSTValidator
        # super: box_loss, cls_loss, dfl_loss, weather_loss, scene_loss, timeofday_loss
        # base:  same
        # kd:    box_kd, cls_kd, dfl_kd, feat_kd
        self.loss_names = (
            "super/box", "super/cls", "super/dfl", "super/weather", "super/scene", "super/timeofday",
            "base/box",  "base/cls",  "base/dfl",  "base/weather",  "base/scene",  "base/timeofday",
            "kd/box",    "kd/cls",    "kd/dfl",    "kd/feat",
        )
        return DetectWSTValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        
    def validate(self, skip=None):
        metrics = self.validator(self, skip=skip)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness
    
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
                    skip_super = [False] * len(skip)
                    preds_super = self.model(batch["img"], skip=skip_super, return_features=True)
 
                    # MTL(Detect + WST) loss for super model
                    loss_super, self.loss_items_super = unwrap_model(self.model).loss(batch, preds=preds_super)
                    self.loss_super = loss_super.sum()
                    if RANK != -1:
                        self.loss_super *= world_size
                    self.tloss_super = (
                        (self.tloss_super * i + self.loss_items_super) / (i + 1)
                        if self.tloss_super is not None else self.loss_items_super
                    )
                    
                    # Backward super
                    with torch.amp.autocast('cuda', enabled=False):
                        self.scaler.scale(self.loss_super).backward()
 
                    # ── Forward base (skip=True) ────────────────────────────
                    skip_base = [True] * len(skip)
                    preds_base = self.model(batch["img"], skip=skip_base, return_features=True)
 
                    # MTL(Detect + WST) loss for base model
                    loss_base, self.loss_items_base = unwrap_model(self.model).loss(batch, preds=preds_base)
                    self.loss_base = loss_base.sum()
                    if RANK != -1:
                        self.loss_base *= world_size
                    self.tloss_base = (
                        (self.tloss_base * i + self.loss_items_base) / (i + 1)
                        if self.tloss_base is not None else self.loss_items_base
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
 
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
 
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
 
                # Validation: super
                print("XXXXXX Validation super XXXXXX")
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate(skip=[False] * len(skip))
                self.save_metrics(metrics={**self.label_loss_items(self.tloss_super, prefix="train/super"), **self.metrics, **self.lr})
 
                # Validation: base
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate(skip=[True] * len(skip))
                self.save_metrics(metrics={**self.label_loss_items(self.tloss_base, prefix="train/base"), **self.metrics, **self.lr})
 
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