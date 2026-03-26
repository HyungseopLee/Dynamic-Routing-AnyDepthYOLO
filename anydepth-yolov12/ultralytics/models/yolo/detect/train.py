# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
import os
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
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


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get batch size by calculating memory occupation of model."""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # 4 for mosaic augmentation
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)



class DetectionTrainerAnyDepth(DetectionTrainer):
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        from ultralytics.models.yolo.detect.val import DetectionValidatorAnyDepth
        self.loss_names = "box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo"
        return DetectionValidatorAnyDepth(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        from ultralytics.nn.tasks import DetectionModelAnyDepth
        model = DetectionModelAnyDepth(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def validate(self, skip=None):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self, skip=skip)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        # if os.getenv("YOLOV12_DETECT_ANOMALY") == "1":
        #     torch.autograd.set_detect_anomaly(True)

        if RANK in {-1, 0}:
            print(self.model)  # print model

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
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
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            
            # XXXXXXXXXXXXXXXXXXXXXXX
            # experimental: update momentum at epoch 500 to 0.9 for SGD optimizer
            # this makes model more agile in the final stage of training
            # if self.epoch >= 500:
            #     target_momentum = 0.9
            #     if any("momentum" in x and abs(x["momentum"] - target_momentum) > 1e-5 for x in self.optimizer.param_groups):
            #         LOGGER.info(f"Updating momentum to {target_momentum} at epoch {self.epoch}")
            #         for x in self.optimizer.param_groups:
            #             if "momentum" in x:
            #                 x["momentum"] = target_momentum

            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                # pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss_super = None
            self.tloss_base = None
            self.tloss_kd = None

            skip = [False,] * unwrap_model(self.model).num_skippable_layers

            for i, batch in enumerate(self.train_loader):
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward super
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    # exp: False for super, True for base
                    skip = [False,] * len(skip)
                    preds_super = self.model(batch["img"], skip=skip, return_features=True)

                    # Extract predictions from dictionary if return_features=True
                    if isinstance(preds_super, dict) and "pred" in preds_super:
                        preds_ = preds_super["pred"]
                    else:
                        preds_ = preds_super
                    loss_super, self.loss_items = unwrap_model(self.model).loss(batch, preds_)

                    self.loss_super = loss_super.sum()
                    if RANK != -1:
                        self.loss_super *= world_size
                    self.tloss_super = (
                        (self.tloss_super * i + self.loss_items) / (i + 1) if self.tloss_super is not None else self.loss_items
                    )

                    with torch.amp.autocast('cuda', enabled=False):
                        # Backward base
                        # print("XXXXXX Backward base XXXXXX")
                        # self.scaler.scale(self.loss).backward(retain_graph=True)
                        self.scaler.scale(self.loss_super).backward()

                
                    # experiment: freeze detection head for base model
                    '''
                    from ultralytics.nn.modules.head import Detect
                    for module in unwrap_model(self.model).modules():
                        if isinstance(module, Detect):
                            for param in module.parameters():
                                param.requires_grad = False
                            # print("XXXXXX Freeze detection head for base model XXXXXX")

                    '''
                    # Forward base
                    # print("XXXXXX Forward super XXXXXX")
                    skip = [True,] * len(skip)
                    preds_base = self.model(batch["img"], skip=skip, return_features=True)
                    # Extract predictions from dictionary if return_features=True
                    if isinstance(preds_base, dict) and "pred" in preds_base:
                        preds_ = preds_base["pred"]
                    else:
                        preds_ = preds_base
                    loss_base, self.loss_items_base = unwrap_model(self.model).loss(batch, preds_)
                    self.loss_base = loss_base.sum()
                    if RANK != -1:
                        self.loss_base *= world_size
                    self.tloss_base = (
                        (self.tloss_base * i + self.loss_items_base) / (i + 1) if self.tloss_base is not None else self.loss_items_base
                    )

                    loss_kd, self.loss_items_kd = unwrap_model(self.model).loss(
                        batch,
                        preds=preds_super,
                        preds_base=preds_base,
                    )
                    self.loss_kd = loss_kd.sum()
                      # print(f"XXXXXX {self.loss} {self.loss_items} XXXXXX")

                    if RANK != -1:
                        self.loss_kd *= world_size
                    self.tloss_kd = (
                        (self.tloss_kd * i + self.loss_items_kd) / (i + 1) if self.tloss_kd is not None else self.loss_items_kd
                    )

                    # warmup 
                    # warmup_epochs = self.args.kd_warmup_epochs if hasattr(self.args, 'kd_warmup_epochs') else 30
                    # kd_scale = 1.0
                    # if epoch < warmup_epochs:
                    #     kd_scale = 0.5 * (1 - math.cos(math.pi * epoch / warmup_epochs))
                    
                    # XXXXXXXXXXXXXXXXXXXXXXX
                    # experimental: reduce kd loss weight by half after epoch 500 
                    # for more agile final training stage
                    # if epoch >= 500:
                    if False:
                        kd_scale = 0.5                        
                    else:
                        kd_scale = 1.0

                    self.loss = self.args.alpha_base * self.loss_base + \
                        (1 - self.args.alpha_base) * self.loss_kd * kd_scale

                        
                                   
                    # self.loss = self.loss_kd
                    with torch.amp.autocast('cuda', enabled=False):
                        # Backward base
                        # print("XXXXXX Backward base XXXXXX")
                        self.scaler.scale(self.loss).backward()
                    '''
                    # unfreeze detection head after base model backward
                    for module in unwrap_model(self.model).modules():
                        if isinstance(module, Detect):
                            for param in module.parameters():
                                param.requires_grad = True
                            # print("XXXXXX Unfreeze detection head for base model XXXXXX")
                    '''


                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss_super.shape[0] if len(self.tloss_super.shape) else 1
                    if i % max(1, nb // 10) == 0:
                        progress_msg = (
                            f"Epoch {epoch + 1}/{self.epochs}, Batch {i + 1}/{nb} "
                            f"({100 * (i + 1) / nb:.1f}%) - "
                            # f"Memory: {self._get_memory():.3g}G, "
                            f"Loss base: {self.loss_base:.3f}, "
                            f"Loss super: {self.loss_super:.3f}, "
                            f"Loss kd: {self.loss_kd:.3f}, "
                            f"base box,cls,dfl: {self.loss_items_base[0]:.3f},{self.loss_items_base[1]:.3f},{self.loss_items_base[2]:.3f}, "
                            f"super box,cls,dfl: {self.loss_items[0]:.3f},{self.loss_items[1]:.3f},{self.loss_items[2]:.3f}, "
                            f"kd box,cls,dfl,feat: {self.loss_items_kd[0]:.5f},{self.loss_items_kd[1]:.5f},{self.loss_items_kd[2]:.5f},{self.loss_items_kd[3]:.5f}"
                        )
                        LOGGER.info(progress_msg)
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                #'''
                print("XXXXXX Validation super XXXXXX")
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate(skip=[False,] * len(skip))
                self.save_metrics(metrics={**self.label_loss_items(self.tloss_super), **self.metrics, **self.lr})
                #'''
                print("XXXXXX Validation base XXXXXX")
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate(skip=[True,] * len(skip))
                self.save_metrics(metrics={**self.label_loss_items(self.tloss_base), **self.metrics, **self.lr})
                #'''                            
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
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
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


