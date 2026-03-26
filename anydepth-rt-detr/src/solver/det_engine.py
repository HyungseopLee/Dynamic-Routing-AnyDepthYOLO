"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 

import torch.nn.functional as F

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

from fvcore.nn import FlopCountAnalysis, flop_count_table

def train_one_epoch_self_distill(model: torch.nn.Module, criterion: torch.nn.Module,
                    criterion_kd: torch.nn.Module,      
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    criterion_kd.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    # i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # i += 1
        # if i == 10:
        #     break

        if scaler is not None:
            assert False, 'Do not support AMP yet'

            # backbone_skip=[False, False, False, False]
            # encoder_skip=[False,]
            # decoder_skip=[False,]
            # # supernet forward
            # with torch.autocast(device_type=str(device), cache_enabled=True):
            #     outputs_super = model(samples, targets, backbone_skip, encoder_skip, decoder_skip)
            
            # outputs_super_logits_topK, outputs_super_logits_topK_idx = outputs_super['pred_logits'].topk(10, dim=2, largest=True, sorted=True)
            # outputs_super_boxes = outputs_super['pred_boxes']

            # with torch.autocast(device_type=str(device), enabled=False):
            #     loss_dict = criterion(outputs_super, targets)

            # loss_full = sum(loss_dict.values())

            # # supernet backward
            # scaler.scale(loss_full).backward()

            # # [TODO] kd loss
            # # .... 
            # backbone_skip=[True, True, True, True]
            # encoder_skip=[True,]
            # decoder_skip=[False,]

            # with torch.autocast(device_type=str(device), cache_enabled=True):
            #     outputs_skip = model(samples, targets, backbone_skip, encoder_skip, decoder_skip)
            
            # outputs_skip_logits_topK = outputs_skip['pred_logits'].gather(2, outputs_super_logits_topK_idx)
            # outputs_skip_boxes = outputs_skip['pred_boxes']
            
            # with torch.autocast(device_type=str(device), enabled=False):
            #     T = 1.0 # temperature
            #     loss_kd_logits = criterion_kd(F.log_softmax(outputs_skip_logits_topK[:,:10]/T, dim=2), F.softmax(outputs_super_logits_topK[:,:10].clone().detach() / T, dim=2)) * T * T
            #     loss_kd_boxes = criterion_kd(F.log_softmax(outputs_skip_boxes/T, dim=2), F.softmax(outputs_super_boxes.clone().detach() / T, dim=2)) * T * T
            
            # alpha = 0.5
            # loss_kd = (1. - alpha) * loss_kd_logits + alpha * loss_kd_boxes

            # # basenet backward
            # scaler.scale(loss_kd).backward()

            # if max_norm > 0:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad()

        else:
            # assert False, 'Not implemented yet'
            
            optimizer.zero_grad()
            
            backbone_skip = [False, False, False, False]
            encoder_skip = [False,]
            decoder_depth = 6
            
            # supernet forward/backward
            outputs_super = model(samples, targets, backbone_skip, encoder_skip, decoder_depth)

            loss_dict = criterion(outputs_super, targets)
            
            loss_full = sum(loss_dict.values())
            # print("loss_full: ", loss_dict.keys())
            loss_full.backward()

            # EXP: basenet forward/backward
            backbone_skip = [True, True, True, True]
            encoder_skip = [True,]
            decoder_depth = 6

            outputs_base = model(samples, targets, backbone_skip, encoder_skip, decoder_depth)

            outputs_super['pred_logits'] = outputs_super['pred_logits'].detach()
            outputs_super['pred_boxes'] = outputs_super['pred_boxes'].detach()
            # aux_output is a list of dictionaries
            # outputs_super['aux_outputs'] = outputs_super['aux_outputs']

            # print("outputs_base keys: ", outputs_base.keys())
            # print("outputs_super keys: ", outputs_super.keys())            
    
            loss_skip_dict = criterion_kd(outputs_base, outputs_super, targets)
            
            # print("loss_skip_dict: ", loss_skip_dict.keys())
            loss_skip = sum(loss_skip_dict.values())
            # print("loss_skip: ", loss_skip_dict.keys())
            loss_skip.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        # print("loss_dict_reduced: ", loss_dict_reduced)
        loss_value = sum(loss_dict_reduced.values())
        
        loss_skip_dict_reduced = reduce_dict(loss_skip_dict)
        loss_skip_dict_reduced_renamed = {}
        for k, v in loss_skip_dict_reduced.items():
            loss_skip_dict_reduced_renamed[k+'_skip'] = v
        # print("loss_skip_dict_reduced: ", loss_skip_dict_reduced_renamed)
        loss_skip_value = sum(loss_skip_dict_reduced_renamed.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_skip_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(loss_skip=loss_skip_value, **loss_skip_dict_reduced_renamed)
        
        batch_size = samples.shape[0]

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# woochul: experiment 2025.02.04
def train_one_epoch_two_backwards(model: torch.nn.Module, criterion: torch.nn.Module,
                    criterion_kd: torch.nn.Module,      
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    criterion_kd.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    # i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # i += 1
        # if i == 10:
        #     break

        if scaler is not None:
            assert False, 'Do not support AMP yet'

            # backbone_skip=[False, False, False, False]
            # encoder_skip=[False,]
            # decoder_skip=[False,]
            # # supernet forward
            # with torch.autocast(device_type=str(device), cache_enabled=True):
            #     outputs_super = model(samples, targets, backbone_skip, encoder_skip, decoder_skip)
            
            # outputs_super_logits_topK, outputs_super_logits_topK_idx = outputs_super['pred_logits'].topk(10, dim=2, largest=True, sorted=True)
            # outputs_super_boxes = outputs_super['pred_boxes']

            # with torch.autocast(device_type=str(device), enabled=False):
            #     loss_dict = criterion(outputs_super, targets)

            # loss_full = sum(loss_dict.values())

            # # supernet backward
            # scaler.scale(loss_full).backward()

            # # [TODO] kd loss
            # # .... 
            # backbone_skip=[True, True, True, True]
            # encoder_skip=[True,]
            # decoder_skip=[False,]

            # with torch.autocast(device_type=str(device), cache_enabled=True):
            #     outputs_skip = model(samples, targets, backbone_skip, encoder_skip, decoder_skip)
            
            # outputs_skip_logits_topK = outputs_skip['pred_logits'].gather(2, outputs_super_logits_topK_idx)
            # outputs_skip_boxes = outputs_skip['pred_boxes']
            
            # with torch.autocast(device_type=str(device), enabled=False):
            #     T = 1.0 # temperature
            #     loss_kd_logits = criterion_kd(F.log_softmax(outputs_skip_logits_topK[:,:10]/T, dim=2), F.softmax(outputs_super_logits_topK[:,:10].clone().detach() / T, dim=2)) * T * T
            #     loss_kd_boxes = criterion_kd(F.log_softmax(outputs_skip_boxes/T, dim=2), F.softmax(outputs_super_boxes.clone().detach() / T, dim=2)) * T * T
            
            # alpha = 0.5
            # loss_kd = (1. - alpha) * loss_kd_logits + alpha * loss_kd_boxes

            # # basenet backward
            # scaler.scale(loss_kd).backward()

            # if max_norm > 0:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad()

        else:
            # assert False, 'Not implemented yet'
            
            optimizer.zero_grad()
            
            backbone_skip = [False, False, False, False]
            encoder_skip = [False,]
            decoder_depth = 6
            
            # supernet forward/backward
            outputs_super = model(samples, targets, backbone_skip, encoder_skip, decoder_depth)

            loss_dict = criterion(outputs_super, targets)
            
            loss_full = sum(loss_dict.values())
            # print("loss_full: ", loss_dict.keys())
            loss_full.backward()

            # EXP: basenet forward/backward
            backbone_skip = [True, True, True, True]
            encoder_skip = [True,]
            decoder_depth = 6

            outputs_base = model(samples, targets, backbone_skip, encoder_skip, decoder_depth)

            loss_skip_dict = criterion(outputs_base, targets)
            
            loss_skip = sum(loss_skip_dict.values())
            # print("loss_full: ", loss_dict.keys())
            loss_skip.backward()




            # outputs_super['pred_logits'] = outputs_super['pred_logits'].detach()
            # outputs_super['pred_boxes'] = outputs_super['pred_boxes'].detach()
            # # aux_output is a list of dictionaries
            # # outputs_super['aux_outputs'] = outputs_super['aux_outputs']

            # print("outputs_base keys: ", outputs_base.keys())
            # print("outputs_super keys: ", outputs_super.keys())            
    
            # loss_skip_dict = criterion_kd(outputs_base, outputs_super, targets)
            
            # print("loss_skip_dict: ", loss_skip_dict.keys())
            # loss_skip = sum(loss_skip_dict.values())
            # # print("loss_skip: ", loss_skip_dict.keys())
            # loss_skip.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        # print("loss_dict_reduced: ", loss_dict_reduced)
        loss_value = sum(loss_dict_reduced.values())
        
        loss_skip_dict_reduced = reduce_dict(loss_skip_dict)
        loss_skip_dict_reduced_renamed = {}
        for k, v in loss_skip_dict_reduced.items():
            loss_skip_dict_reduced_renamed[k+'_skip'] = v
        # print("loss_skip_dict_reduced: ", loss_skip_dict_reduced_renamed)
        loss_skip_value = sum(loss_skip_dict_reduced_renamed.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_skip_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(loss_skip=loss_skip_value, **loss_skip_dict_reduced_renamed)
        
        batch_size = samples.shape[0]

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_adn(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir,
             backbone_skip=[False, False, False, False], encoder_skip=[False,], decoder_depth=6):
    model.eval()
    
    # @HyungseopLee: Logging FLOPs with current depth config
    print("=" * 100)
    print(f"Start evaluate!", flush=True)
    print(f"Depth config - Backbone: {backbone_skip}, Encoder: {encoder_skip}, Decoder: {decoder_depth}")
    model.to(device)
    class FLOPsWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            return self.base_model(
                x, 
                backbone_skip=backbone_skip, 
                encoder_skip=encoder_skip, 
                decoder_depth=decoder_depth
            )
    sample_batch = next(iter(data_loader))[0] 
    h, w = sample_batch.shape[-2:] # (B, C, H, W) -> H, W
    input_tensor = torch.randn(1, 3, 640, 640).to(device)
    print(f"input_tensor.shape: {input_tensor.shape}")
    flops = FlopCountAnalysis(FLOPsWrapper(model), input_tensor)
    print(flop_count_table(flops))
    
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, backbone_skip=backbone_skip, encoder_skip=encoder_skip, decoder_depth=decoder_depth)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    # i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # i += 1
        # if i == 10:
        #     break

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    
    model.eval()
    
    # @HyungseopLee: Logging FLOPs with current depth config
    print("=" * 100)
    print(f"Start evaluate!", flush=True)
    model.to(device)
    sample_batch = next(iter(data_loader))[0] 
    h, w = sample_batch.shape[-2:] # (B, C, H, W) -> H, W
    input_tensor = torch.randn(1, 3, 640, 640).to(device)
    print(f"input_tensor.shape: {input_tensor.shape}")
    flops = FlopCountAnalysis(model, input_tensor)
    print(flop_count_table(flops))
    
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)


    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            

    return stats, coco_evaluator
