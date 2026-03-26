"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register


# @register
# class SetCriterionKD2(nn.Module):
#     """ This class computes KD loss for RTDETR
#     """
#     __share__ = ['num_classes', ]
#     __inject__ = ['matcher', ]

#     def __init__(self, matcher, weight_dict_kd, weight_dict_aux, losses, losses_aux, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
#         """ Create the criterion.
#         Parameters:
#             num_classes: number of object categories, omitting the special no-object category
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losse  and as values their relative weight.
#             eos_coef: relative classification weight applied to the no-object category
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict_kd = weight_dict_kd
#         self.weight_dict_aux = weight_dict_aux
#         self.losses = losses 
#         self.losses_aux = losses_aux

#         empty_weight = torch.ones(self.num_classes + 1)
#         empty_weight[-1] = eos_coef
#         self.register_buffer('empty_weight', empty_weight)

#         self.alpha = alpha
#         self.gamma = gamma

#     def loss_features_kd(self, outputs, outputs_super, targets, indices, num_boxes, log=True):
#         losses = {}
#         features =[]
#         for i, intermediate_output in enumerate(outputs['intermediate_backbone']):
#             # print(f"[{i}]: {intermediate_output.shape}")    
#             xx = F.adaptive_avg_pool2d(intermediate_output, (1,1))
#             # print(f"xx shape: {xx.shape}")
#             features.append(xx)
#         features_backbone_skip = torch.squeeze(torch.concat(features, dim=1))

#         features = []
#         for i, intermediate_output in enumerate(outputs_super['intermediate_backbone']):
#             # print(f"[{i}]: {intermediate_output.shape}")    
#             xx = F.adaptive_avg_pool2d(intermediate_output, (1,1))
#             # print(f"xx shape: {xx.shape}")
#             features.append(xx)
#         features_backbone = torch.squeeze(torch.concat(features, dim=1))

#         loss = F.kl_div(F.log_softmax(features_backbone_skip, dim=-1), F.softmax(features_backbone.clone().detach(), dim=-1), reduction='batchmean')  
#         losses['loss_features_backbone_kd'] = loss

#         features =[]
#         for i, intermediate_output in enumerate(outputs['intermediate_encoder']):
#             # print(f"[{i}]: {intermediate_output.shape}")    
#             xx = F.adaptive_avg_pool2d(intermediate_output, (1,1))
#             # print(f"xx shape: {xx.shape}")
#             features.append(xx)
#         features_encoder_skip = torch.squeeze(torch.concat(features, dim=1))

#         features = []
#         for i, intermediate_output in enumerate(outputs_super['intermediate_encoder']):
#             # print(f"[{i}]: {intermediate_output.shape}")    
#             xx = F.adaptive_avg_pool2d(intermediate_output, (1,1))
#             # print(f"xx shape: {xx.shape}")
#             features.append(xx)
#         features_encoder = torch.squeeze(torch.concat(features, dim=1))

#         loss = F.kl_div(F.log_softmax(features_encoder_skip, dim=-1), F.softmax(features_encoder.clone().detach(), dim=-1), reduction='batchmean')  
        
#         losses['loss_features_encoder_kd'] = loss
#         return losses 
        
    
#     def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)

#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
#         ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
#         ious = torch.diag(ious).detach()

#         src_logits = outputs['pred_logits']
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o
#         target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

#         target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
#         target_score_o[idx] = ious.to(target_score_o.dtype)
#         target_score = target_score_o.unsqueeze(-1) * target

#         pred_score = F.sigmoid(src_logits).detach()
#         weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
#         loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
#         loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
#         return {'loss_vfl': loss}

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

#         losses = {}

#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes

#         loss_giou = 1 - torch.diag(generalized_box_iou(
#                 box_cxcywh_to_xyxy(src_boxes),
#                 box_cxcywh_to_xyxy(target_boxes)))
#         losses['loss_giou'] = loss_giou.sum() / num_boxes
#         return losses


#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss_kd(self, loss, outputs, outputs_super, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'boxes': self.loss_boxes,        
#             'vfl': self.loss_labels_vfl,
#             'features_kd': self.loss_features_kd,
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         if loss == 'features_kd':
#             return loss_map[loss](outputs, outputs_super, targets, indices, num_boxes, **kwargs)
#         else:
#             return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'boxes': self.loss_boxes,
#             'vfl': self.loss_labels_vfl,
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

#     def forward(self, outputs, outputs_super, targets):
#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
#         """
#         # outputs_without_aux = {k: v for k, v in outputs_super.items() if 'aux' not in k}
#         outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

#         # Retrieve the matching between the outputs of the last layer and the targets
#         indices = self.matcher(outputs_without_aux, targets)

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#         if is_dist_available_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             l_dict = self.get_loss_kd(loss, outputs, outputs_super, targets, indices, num_boxes)
#             l_dict = {k: l_dict[k] * self.weight_dict_kd[k] for k in l_dict if k in self.weight_dict_kd}
#             losses.update(l_dict)
    
#         # # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for i, aux_outputs in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_outputs, targets)
#                 for loss in self.losses_aux:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}

#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k: l_dict[k] * self.weight_dict_aux[k] for k in l_dict if k in self.weight_dict_aux}
#                     l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#                     # print("[super] losses keys #2: ", losses.keys())
 
#         # In case of cdn auxiliary losses. For rtdetr
#         if 'dn_aux_outputs' in outputs:
#             assert 'dn_meta' in outputs, ''
#             indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
#             num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

#             for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
#                 # indices = self.matcher(aux_outputs, targets)
#                 for loss in self.losses_aux:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}

#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k: l_dict[k] * self.weight_dict_aux[k] for k in l_dict if k in self.weight_dict_aux}
#                     l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)
              
#         return losses
    
#     @staticmethod
#     def get_cdn_matched_indices(dn_meta, targets):
#         '''get_cdn_matched_indices
#         '''
#         dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
#         num_gts = [len(t['labels']) for t in targets]
#         device = targets[0]['labels'].device
        
#         dn_match_indices = []
#         for i, num_gt in enumerate(num_gts):
#             if num_gt > 0:
#                 gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
#                 gt_idx = gt_idx.tile(dn_num_group)
#                 assert len(dn_positive_idx[i]) == len(gt_idx)
#                 dn_match_indices.append((dn_positive_idx[i], gt_idx))
#             else:
#                 dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
#                     torch.zeros(0, dtype=torch.int64,  device=device)))
        
#         return dn_match_indices



@register
class SetCriterionKD(nn.Module):
    """ This class computes KD loss for RTDETR
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict_kd, weight_dict_dn, losses_kd, losses_kd_aux, losses_dn, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losse  and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict_kd = weight_dict_kd
        self.weight_dict_dn = weight_dict_dn
        self.losses_kd = losses_kd 
        self.losses_kd_aux = losses_kd_aux
        self.losses_dn = losses_dn
    
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

    def loss_features_kd(self, outputs, outputs_super, targets, indices, indices_super, num_boxes, log=True):
        losses = {}
        features =[]
        for i, intermediate_output in enumerate(outputs['intermediate_backbone']):
            # print(f"[{i}]: {intermediate_output.shape}")    
            xx = F.adaptive_avg_pool2d(intermediate_output, (1,1))
            # print(f"xx shape: {xx.shape}")
            features.append(xx)
        features_backbone_skip = torch.squeeze(torch.concat(features, dim=1))

        features = []
        for i, intermediate_output in enumerate(outputs_super['intermediate_backbone']):
            # print(f"[{i}]: {intermediate_output.shape}")    
            xx = F.adaptive_avg_pool2d(intermediate_output, (1,1))
            # print(f"xx shape: {xx.shape}")
            features.append(xx)
        features_backbone = torch.squeeze(torch.concat(features, dim=1))

        loss = F.kl_div(F.log_softmax(features_backbone_skip, dim=-1), F.softmax(features_backbone.clone().detach(), dim=-1), reduction='batchmean')  
        losses['loss_features_backbone_kd'] = loss

        features =[]
        for i, intermediate_output in enumerate(outputs['intermediate_encoder']):
            # print(f"[{i}]: {intermediate_output.shape}")    
            xx = F.adaptive_avg_pool2d(intermediate_output, (1,1))
            # print(f"xx shape: {xx.shape}")
            features.append(xx)
        features_encoder_skip = torch.squeeze(torch.concat(features, dim=1))

        features = []
        for i, intermediate_output in enumerate(outputs_super['intermediate_encoder']):
            # print(f"[{i}]: {intermediate_output.shape}")    
            xx = F.adaptive_avg_pool2d(intermediate_output, (1,1))
            # print(f"xx shape: {xx.shape}")
            features.append(xx)
        features_encoder = torch.squeeze(torch.concat(features, dim=1))

        loss = F.kl_div(F.log_softmax(features_encoder_skip, dim=-1), F.softmax(features_encoder.clone().detach(), dim=-1), reduction='batchmean')  
        
        losses['loss_features_encoder_kd'] = loss
        return losses 
        
    def loss_labels_vfl_kd(self, outputs, outputs_super, targets, indices, indices_super, num_boxes, log=True):
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        idx_super = self._get_src_permutation_idx(indices_super)

        src_logits = outputs['pred_logits']
        target_logits = outputs_super['pred_logits'].clone().detach()

        # exp: only use valid logits
        # print(src_logits.shape, idx.shape)
        loss = F.kl_div(F.log_softmax(src_logits[idx], dim=-1), F.softmax(target_logits[idx_super], dim=-1), reduction='none')  
        
        # exp: weight might not right for kl_div....
        # loss = loss * weight        
        # print("loss negatives, positives: ", (loss < 0).sum(), (loss > 0).sum())
        # 
        # print("loss.shape: ", loss.shape)
            
        # exp #2: cross_entropy
        # loss = F.binary_cross_entropy_with_logits(src_logits, F.softmax(target_logits, dim=-1), weight=weight, reduction='none')
        # print("loss negatives, positives: ", (loss < 0).sum(), (loss > 0).sum())

        # exp #3: original cross_entropy between src_logits and target_score
        # loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        # print("loss.shape: ", loss.shape)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        # print("loss", loss)

        return {'loss_vfl_kd': loss}
    

    # def loss_labels_vfl_kd_old(self, outputs, outputs_super, targets, indices, indices_super, num_boxes, log=True):
    #     assert 'pred_boxes' in outputs
    #     idx = self._get_src_permutation_idx(indices)
    #     idx_super = self._get_src_permutation_idx(indices_super)

    #     src_boxes = outputs['pred_boxes'][idx]
    #     target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    #     ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
    #     ious = torch.diag(ious).detach()

    #     src_logits = outputs['pred_logits']
    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    #     target_classes = torch.full(src_logits.shape[:2], self.num_classes,
    #                                 dtype=torch.int64, device=src_logits.device)
    #     target_classes[idx] = target_classes_o
    #     target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

    #     target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
    #     target_score_o[idx] = ious.to(target_score_o.dtype)
    #     target_score = target_score_o.unsqueeze(-1) * target

    #     pred_score = F.sigmoid(src_logits).detach()
    #     weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score


    #     # print("weight:", weight[0, :10, :10])
    #     # print("negative weight:", (weight < 0).sum())
    #     # print("weight.shape: ", weight.shape)
    #     # weight = weight.sum(2)
    #     # print("weight.shape: ", weight.shape)
    #     # print("weight min max: ", weight.min(), weight.max())

    #     target_logits = outputs_super['pred_logits']

    #     # print("src_logits min, max: ", src_logits.min(), src_logits.max())
    #     # print("target_logits min, max: ", target_logits.min(), target_logits.max())

    #     # exp #1: kl_div
    #     # loss = F.kl_div(F.log_softmax(src_logits, dim=-1), F.softmax(target_logits, dim=-1), reduction='none')  

    #     # exp: only use valid logits
    #     # print(src_logits.shape, idx.shape)
    #     loss = F.kl_div(F.log_softmax(src_logits[idx], dim=-1), F.softmax(target_logits[idx_super], dim=-1), reduction='none')  
        
    #     # exp: weight might not right for kl_div....
    #     # loss = loss * weight        
    #     # print("loss negatives, positives: ", (loss < 0).sum(), (loss > 0).sum())
    #     # 
    #     # print("loss.shape: ", loss.shape)
            
    #     # exp #2: cross_entropy
    #     # loss = F.binary_cross_entropy_with_logits(src_logits, F.softmax(target_logits, dim=-1), weight=weight, reduction='none')
    #     # print("loss negatives, positives: ", (loss < 0).sum(), (loss > 0).sum())

    #     # exp #3: original cross_entropy between src_logits and target_score
    #     # loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
    #     # print("loss.shape: ", loss.shape)
    #     loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
    #     # print("loss", loss)

    #     # loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
    #     # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
    #     return {'loss_vfl_kd': loss}
    

    def loss_boxes_kd(self, outputs, outputs_super, targets, indices, indices_super, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        idx_super = self._get_src_permutation_idx(indices_super)
        target_boxes = outputs_super['pred_boxes'].clone().detach()[idx_super] # EXP...

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox_kd'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou_kd'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss_kd(self, loss, outputs, outputs_super, targets, indices, indices_super, num_boxes, **kwargs):
        loss_map = {
            'boxes_kd': self.loss_boxes_kd,        
            'vfl_kd': self.loss_labels_vfl_kd,
            'features_kd': self.loss_features_kd,
        }
        # print("get_loss_kd: ", loss)
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, outputs_super, targets, indices, indices_super, num_boxes, **kwargs)
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def _sort_indices_super(self, indices, indices_super):
        # woochul: sort outputs_super by indices of outputs
        for i in range(len(indices)): # for each batch
            # 
            sorted_index = torch.argsort(indices_super[i][1])
            # 
            new_index = sorted_index[indices[i][1]]
            indices_super[i] = (indices_super[i][0][new_index], new_index)
        return indices_super

    def forward(self, outputs, outputs_super, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # wrong(?)
        # outputs_without_aux = {k: v for k, v in outputs_super.items() if 'aux' not in k}
        # fixed
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        outputs_super_without_aux = {k: v for k, v in outputs_super.items() if 'aux' not in k}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices_super = self.matcher(outputs_super_without_aux, targets)
        
        # sort outputs_super by indices of outputs
        # for i in range(len(indices)):
        #     sorted_index = torch.argsort(indices_super[i][1])
        #     new_index = sorted_index[indices[i][1]]
        #     # print(new_index.shape)
        #     indices_super[i] = (indices_super[i][0][new_index], new_index)
        indices_super = self._sort_indices_super(indices, indices_super)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses_kd:
            l_dict = self.get_loss_kd(loss, outputs, outputs_super, targets, indices, indices_super, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict_kd[k] for k in l_dict if k in self.weight_dict_kd}
            losses.update(l_dict)
    
        # # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # print(f"aux_outputs #{i}")
                indices = self.matcher(aux_outputs, targets)
                
                # exp #1: distill from super's corresponding aux_output
                # aux_outputs_super = outputs_super['aux_outputs'][i]
                # indices_super = self.matcher(aux_outputs_super, targets)
                # indices_super = self._sort_indices_super(indices, indices_super)

                
                # for loss in self.losses_kd_aux:  # losses_kd_aux does not have 'features_kd'
                for loss in self.losses_dn: # fix this to losses_aux  
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    # original
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    
                    # exp #1
                    # l_dict = self.get_loss_kd(loss, aux_outputs, aux_outputs_super, targets, indices, indices_super, num_boxes, **kwargs)

                    # exp #2: distill from super's last output
                    # l_dict = self.get_loss_kd(loss, aux_outputs, outputs_super, targets, indices, indices_super, num_boxes, **kwargs)
                    # l_dict = {k: l_dict[k] * self.weight_dict_kd[k] for k in l_dict if k in self.weight_dict_kd}
                    l_dict = {k: l_dict[k] * self.weight_dict_dn[k] for k in l_dict if k in self.weight_dict_dn}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    # print("losses keys #1: ", l_dict.keys())
                    losses.update(l_dict)
                    # print("[super] losses keys #2: ", losses.keys())
 
        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses_dn:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict_dn[k] for k in l_dict if k in self.weight_dict_dn}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
              
        return losses
    
    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices


@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losse  and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
           
        # # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    # print("[super] losses keys #2: ", losses.keys())


        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
           

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




