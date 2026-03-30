# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils import RANK

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # keep target_scores_sum as a tensor and ensure >= 1.0 to avoid division by zero
        target_scores_sum = target_scores.sum().clamp_min(1.0)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]



class DetectionLossAnyDepth:
    def __init__(self, model, tal_topk=10):

        self.device = next(model.parameters()).device  # get model device
        m = model.model[-1]  # Detect() module
        self.stride = m.stride 

        self.nc = m.nc # number of classes (e.g., 80)
        self.no = m.no  # number of outputs (e.g., 85)

        self.use_dfl = m.reg_max > 1
        self.reg_max = m.reg_max # DFL max value (e.g., 16)
        self.bbox_loss = BboxLoss(m.reg_max).to(self.device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=self.device)

        hyp = model.args

        self.kd_temp_cls = hyp.kd_temp_cls  # Temperature for classification KL divergence loss
        self.kd_temp_dfl = hyp.kd_temp_dfl  # Temperature for DFL KD loss
        self.kd_temp_feat = hyp.kd_temp_feat  # Temperature for feature KD loss

        self.kd_weight_cls = hyp.kd_cls # Weight for classification KL divergence loss
        self.kd_weight_dfl = hyp.kd_dfl  # Weight for DFL KD loss
        self.kd_weight_box = hyp.kd_box  # Weight for bbox KD loss
        self.kd_weight_feat = hyp.kd_feat  # Weight for feature KL divergence loss

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)

        # debug
        # print(f"kd_weight_cls: {self.kd_weight_cls},"
        #       f"kd_weight_dfl: {self.kd_weight_dfl},"
        #       f"kd_weight_box: {self.kd_weight_box},"
        #       f"kd_weight_feat: {self.kd_weight_feat}");

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    

    def get_vlr_mask(self, fg_mask, anchor_points, gt_bboxes, stride, mask_gt=None, dilation=1):
        """
        fg_mask: [Batch, Num_Anchors] - The binary mask from TAL
        anchor_points: [Num_Anchors, 2] - The (x, y) coordinates of grid centers or pixels
        gt_bboxes: [Batch, Max_Objects, 4] - Ground Truth boxes in xyxy
        stride: [Num_Anchors, 1] - Stride for each anchor
        """
        # 1. Expand GT boxes slightly to create a 'halo'
        pad = dilation * stride  
        
        # 2. Check which anchors fall inside this expanded box
        anchors_x, anchors_y = anchor_points.split(1, dim=-1)
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bboxes.split(1, dim=-1)

        # Shapes for broadcasting:
        # gt components: [Batch, M, 1]
        # anchor terms:  [1, 1, Num_Anchors]
        
        ax_plus_pad = (anchors_x + pad).view(1, 1, -1)
        ax_minus_pad = (anchors_x - pad).view(1, 1, -1)
        ay_plus_pad = (anchors_y + pad).view(1, 1, -1)
        ay_minus_pad = (anchors_y - pad).view(1, 1, -1)
        
        in_outer = (gt_x1 <= ax_plus_pad) & (gt_x2 >= ax_minus_pad) & \
                 (gt_y1 <= ay_plus_pad) & (gt_y2 >= ay_minus_pad)
        
        in_inner = (gt_x1 <= ax_minus_pad) & (gt_x2 >= ax_plus_pad) & \
                 (gt_y1 <= ay_minus_pad) & (gt_y2 >= ay_plus_pad)

        in_vlr = in_outer & (~in_inner)

        if mask_gt is not None:
            if mask_gt.dtype != torch.bool:
                mask_gt = mask_gt.bool()
            in_vlr &= mask_gt

        # 3. VLR is (Anchors in expanded box) MINUS (Anchors already in FG)
        # in_vlr shape: [Batch, M, Num_Anchors] -> any(1) -> [Batch, Num_Anchors]
        vlr_mask = in_vlr.any(dim=1) & (~fg_mask)
        return vlr_mask
        
    def __call__(self, preds_base_dict, preds_super_dict, batch):

        preds_base = preds_base_dict["pred"]
        preds_super = preds_super_dict["pred"]

        # Preprocess super predictions
        assert not isinstance(preds_super, tuple), "preds_super should not be a tuple"
        # bug fix: detach teacher predictions to avoid computing gradients
        pred_super_distri, pred_super_scores = torch.cat([xi.detach().view(preds_super[0].shape[0], self.no, -1) for xi in preds_super], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_super_scores = pred_super_scores.permute(0, 2, 1).contiguous()
        pred_super_distri = pred_super_distri.permute(0, 2, 1).contiguous()

        # Preprocess base predictions
        assert not isinstance(preds_base, tuple), "preds_base should not be a tuple"
        pred_base_distri, pred_base_scores = torch.cat([xi.view(preds_base[0].shape[0], self.no, -1) for xi in preds_base], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_base_scores = pred_base_scores.permute(0, 2, 1).contiguous() # (b, h*w, nc)
        pred_base_distri = pred_base_distri.permute(0, 2, 1).contiguous() # (b, h*w, 4)

        dtype = pred_base_scores.dtype
        batch_size = pred_base_scores.shape[0]
        imgsz = torch.tensor(preds_base[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(preds_base, self.stride, 0.5)
       
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_base_bboxes = self.bbox_decode(anchor_points, pred_base_distri)  # xyxy, (b, h*w, 4)
        pred_super_bboxes = self.bbox_decode(anchor_points, pred_super_distri)  # xyxy, (b, h*w, 4)

        # Assignment between teacher predictions and ground truth
        _, _, target_scores_super, fg_mask_super, target_gt_idx_super = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_super_scores.detach().sigmoid(),
            (pred_super_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # option #2: Assignment between student predictions and ground truth
        _, _, target_scores_base, fg_mask_base, target_gt_idx_base = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_base_scores.detach().sigmoid(),
            (pred_base_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # BG anchors may have arbitrary target_gt_idx values,
        # so we only consider anchors that are FG in both teacher and student for conflict detection
        conflict_mask = (fg_mask_super & fg_mask_base) & (target_gt_idx_super != target_gt_idx_base) # [B, Num_Anchors]

        # if RANK in (0, -1):
        if False:
            print("[debug] fg_mask_super.sum(): ", fg_mask_super.sum().item())
            print("[debug] fg_mask_base.sum(): ", fg_mask_base.sum().item())
            print("[debug] fg_mask common sum: ", (fg_mask_super & fg_mask_base).sum().item())
            print("[debug] conflict_mask.sum(): ", conflict_mask.sum().item())

        # shaprs:
        # pred_super_scores: (B, Num_Anchors, C)
        # pred_base_scores:  (B, Num_Anchors, C)
        # target_scores:     (B, Num_Anchors, C)
        # fg_mask:          (B, Num_Anchors)
        # conflict_mask:    (B, Num_Anchors)
           
        loss = torch.zeros(3, device=self.device)  # kd box, cls, dfl

        # vlr_mask = self.get_vlr_mask(fg_mask, anchor_points * stride_tensor, gt_bboxes, stride_tensor, mask_gt, dilation=1) # [B, Num_Anchors]
        
        loss[1]  = self._classification_kl_loss(
            pred_super_scores.detach().view(-1, self.nc), 
            pred_base_scores.view(-1, self.nc),
            fg_mask_base.view(-1).detach(),
            target_scores_base.detach(), 
            fg_mask_super.view(-1).detach(),
            target_scores_super.detach(),
            conflict_mask.view(-1).detach()
        )
        
        if fg_mask_super.sum() and fg_mask_base.sum():
            loss[0], loss[2] = self.bbox_loss_with_weights(
                pred_super_distri.detach(),  # [B, Num_Anchors, 4*reg_max]
                pred_base_distri,
                pred_super_bboxes.detach(), # [B, Num_Anchors, 4]
                pred_base_bboxes,
                fg_mask_super.detach(), # [B, Num_Anchors]
                fg_mask_base.detach(),
                target_scores_super.detach(),
                target_scores_base.detach(),
                conflict_mask.detach()
            )

        loss[0] *= self.kd_weight_box  # box gain
        loss[1] *= self.kd_weight_cls  # cls gain
        loss[2] *= self.kd_weight_dfl  # dfl gain   
        
        total_kl_loss = loss.sum()

        # Feature loss for knowledge distillation
        if preds_super_dict['features'] is not None and preds_base_dict['features'] is not None:
            # note: preds_base_dict['features'] is used as a teacher distillation target here
            feat_kl_loss = self.kd_weight_feat * self._feature_kl_loss(preds_super_dict['features'], preds_base_dict['features'])
            total_kl_loss = total_kl_loss + feat_kl_loss
        else:
            feat_kl_loss = torch.tensor(0.0, device=self.device)

        kl_loss_items = torch.cat((
            loss.detach(),
            feat_kl_loss.detach().unsqueeze(0)
        ))

        batch_size = preds_base[0].shape[0]  

        return total_kl_loss * batch_size, kl_loss_items


    def _feature_kl_loss(self, teacher_feat, student_feat):
        # features from layers backbone [P4, P6, P8], neck [P11, P14, P17, P20]
        # layer type [C3k2, A2C2f, A2C2f] [A2C2f, A2C2f, A2C2f, C3k2]

        feat_idx = [1, 2, 4, 5, 6] 

        # option #1: concatenate features and compute KL divergence
        # feat_idx_backbone = [1, 2]
        # features_backbone_super = torch.cat([teacher_feat[i] for i in feat_idx_backbone], dim=1)  # [B, C_total]
        # features_backbone_base = torch.cat([student_feat[i] for i in feat_idx_backbone], dim=1)  # [B, C_total]
        # loss_feat_backbone = F.kl_div(F.log_softmax(features_backbone_base / self.kd_temp_feat, dim=-1), \
        #                               F.softmax(features_backbone_super.detach() / self.kd_temp_feat, dim=-1), reduction='batchmean')  

        # feat_idx_neck = [3, 4]
        # features_neck_super = torch.cat([teacher_feat[i] for i in feat_idx_neck], dim=1)  # [B, C_total]
        # features_neck_base = torch.cat([student_feat[i] for i in feat_idx_neck], dim=1)  # [B, C_total]
        # loss_feat_neck = F.kl_div(F.log_softmax(features_neck_base / self.kd_temp_feat, dim=-1), \
        #                           F.softmax(features_neck_super.detach() / self.kd_temp_feat, dim=-1), reduction='batchmean')  

        # option #1: Average backbone and neck feature losses
        # loss_feat = (loss_feat_backbone + loss_feat_neck) / 2.0
        # option #2: backbone feature loss only
        # loss_feat = loss_feat_backbone


        # option #2: compute KL divergence for each feature and average
        # kl_losses = []
        # for i in feat_idx:
        #     teacher_f = teacher_feat[i]  # [B, C_i]
        #     student_f = student_feat[i]  # [B, C_i]
            
        #     # KL divergence for this feature
        #     teacher_probs = F.softmax(teacher_f / self.kd_temp_feat, dim=-1)
        #     student_log_probs = F.log_softmax(student_f / self.kd_temp_feat, dim=-1)
        #     kl_loss_i = F.kl_div(student_log_probs, teacher_probs.detach(), reduction='batchmean')
        #     kl_losses.append(kl_loss_i)
        
        # # Average across all selected features
        # loss_feat = torch.stack(kl_losses).mean()

        # return loss_feat * (self.kd_temp_feat ** 2)

        # Normalized L2 (cosine distance) per layer and average.
        # F.normalize over dim=-1 produces unit vectors; MSE then equals 2*(1 - cosine_sim),
        # which is scale-invariant and correct for avg-pooled feature vectors.
        losses = []
        for i in feat_idx:
            t = teacher_feat[i].detach()                  # [B, C]
            s = student_feat[i]                           # [B, C]
            t_norm = F.normalize(t, dim=-1)               # [B, C]
            s_norm = F.normalize(s, dim=-1)               # [B, C]
            losses.append((s_norm - t_norm).pow(2).sum(dim=-1).mean())

        return sum(losses) / len(losses)


    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)


    def _classification_kl_loss(
        self, 
        teacher_cls, 
        student_cls, 
        fg_mask_base, 
        target_scores_base, 
        fg_mask_super, 
        target_scores_super,
        conflict_mask
    ):        
        teacher_probs = F.softmax(teacher_cls / self.kd_temp_cls, dim=-1)
        student_log_probs = F.log_softmax(student_cls / self.kd_temp_cls, dim=-1)

        kl_per_anchor = F.kl_div(student_log_probs, teacher_probs.detach(), reduction='none').sum(dim=-1)  # [B * Num_Anchors] 
        
        # Teacher-centric: only distill where super is FG and 
        # and student is not in conflict with teacher (i.e., not assigned to a different GT object than teacher)
        fg_mask = fg_mask_super.view(-1) & ~conflict_mask

        # Option A: unweighted (simpler, consistent with GT cls loss pattern)
        # kl_loss = kl_per_anchor[fg_mask].mean()

        # Option B: weighted by teacher quality (consistent with IoU/DFL)
        fg_weight = target_scores_super.sum(-1).view(-1)[fg_mask].detach()
        kl_loss = (kl_per_anchor[fg_mask] * fg_weight).sum() / fg_weight.sum().clamp_min(1)

        return kl_loss * (self.kd_temp_cls ** 2)


    # def _classification_kl_loss(self, teacher_cls, student_cls, fg_mask, vlr_mask, target_scores, target_scores_sum):
    #     teacher_probs = F.softmax(teacher_cls / self.kd_temp_cls, dim=-1)
    #     student_log_probs = F.log_softmax(student_cls / self.kd_temp_cls, dim=-1)

    #     # option #1: only for fg anchors and normalize by total quality score
    #     weight = target_scores.sum(-1).view(-1).detach() # [B * Num_Anchors]
    #     kl_loss_ = F.kl_div(student_log_probs, teacher_probs.detach(), reduction='none').sum(dim=-1) * weight  # [B * Num_Anchors] 
    #     kl_loss = kl_loss_.sum()/ target_scores_sum  # normalize by total quality score of fg anchors

    #     # option #2: include hard bg anchors
    #     # weight_1 = student_cls.sigmoid().max(dim=-1, keepdim=False)[0].detach()  # [B * Num_Anchors] 
    #     # bg_mask_hard = (~fg_mask) & (weight_1 > 0.05)  # mask for hard bg anchors
    #     # weight_1 = weight_1 * bg_mask_hard.float()  # zero out easy bg anchors
    #     # weight_1 = weight_1 * vlr_mask.float()  # halo anchors only  
    #     # weight_2 = target_scores.sum(-1).view(-1) # [B * Num_Anchors]
    #     # weight = torch.max(weight_1, weight_2)  # [B * Num_Anchors]
    #     # kl_loss_ = F.kl_div(student_log_probs, teacher_probs.detach(), reduction='none').sum(dim=-1) * weight.detach()  # [B *Num_Anchors] 
    #     # kl_loss = kl_loss_.sum()/ (weight.sum() + 1e-6)
    #     # if RANK in (-1, 0):
    #     if False:
    #         print(f"[debug][_classification_kl_loss] num_bg_hard: {bg_mask_hard.sum().item()}")
    #         print(f"[debug][_classification_kl_loss] num vlr anchors: {vlr_mask.sum().item()}")
    #         print(f"[debug][_classification_kl_loss] weight_1 sum: {weight_1.sum().item()}")
    #         print(f"[debug][_classification_kl_loss] num_fg: {fg_mask.sum().item()}")
    #         print(f"[debug][_classification_kl_loss] weight_2 sum: {weight_2.sum().item()}")
    #     return kl_loss * (self.kd_temp_cls ** 2)

    def bbox_loss_with_weights(
            self, 
            teacher_dfl, student_dfl, 
            teacher_bbox, student_bbox, 
            fg_mask_super, fg_mask_base,
            target_scores_super, target_scores_base,
            conflict_mask
        ):        
        # experiment: 02.10.2026
        # Use the intersection of fg_mask from teacher and student for iou and dfl loss computation
        # since more stable and less noisy targets can be obtained this way

        # iou loss
        fg_mask = fg_mask_super & fg_mask_base # [B, Num_Anchors]
        # experiment: 
        # ##################
        fg_mask = fg_mask & ~conflict_mask # exclude conflicting anchors from loss computation

        # original
        iou_weight = target_scores_super.sum(-1)[fg_mask].unsqueeze(-1).detach()  # [num_fg, 1]
        iou = bbox_iou(student_bbox[fg_mask], teacher_bbox[fg_mask], xywh=False, CIoU=True)
        iou_loss = ((1.0 - iou) * iou_weight).sum() / iou_weight.sum().clamp_min(1.0)

        # experiment: no weighting, only mask out conflicting anchors
        # iou = bbox_iou(student_bbox[fg_mask], teacher_bbox[fg_mask], xywh=False, CIoU=True)
        # iou_loss = (1.0 - iou).mean()  # average over all non-conflicting fg anchors

        # DFL KL divergence loss
        # Expand fg_mask to cover all 4 coordinates per anchor: [B, Num_Anchors] -> [B * Num_Anchors * 4]
        fg_mask_expanded = fg_mask.unsqueeze(-1).expand(-1, -1, 4).reshape(-1)  # [B * Num_Anchors * 4]
        
        teacher_dfl_fg = teacher_dfl.view(-1, self.reg_max)[fg_mask_expanded]  # [num_fg * 4, reg_max]
        student_dfl_fg = student_dfl.view(-1, self.reg_max)[fg_mask_expanded]  # [num_fg * 4, reg_max]

        # Apply temperature and softmax/log_softmax 
        teacher_probs = F.softmax(teacher_dfl_fg / self.kd_temp_dfl, dim=-1)
        student_log_probs = F.log_softmax(student_dfl_fg / self.kd_temp_dfl, dim=-1)
        
        # Compute KL divergence 
        kl_per_coord = F.kl_div(student_log_probs, teacher_probs.detach(), reduction='none').sum(dim=-1)  # [num_fg * 4]
        kl_per_pixel = kl_per_coord.view(-1, 4).mean(dim=-1)  # [num_fg], mean over 4 coords

        # original weighting: weight by teacher's confidence scores
        dfl_weight = target_scores_super.sum(-1)[fg_mask].detach()  # [num_fg]
        dfl_loss = (kl_per_pixel * dfl_weight).sum() / dfl_weight.sum().clamp_min(1.0)
     
        # no weighting, only mask out conflicting anchors
        # dfl_loss = kl_per_pixel.sum() / fg_mask.sum().clamp_min(1.0)  # average over all non-conflicting fg anchors
        # dfl_loss = dfl_loss * (self.kd_temp_dfl ** 2)  # scale by temperature squared

        return iou_loss, dfl_loss
    
    
# @HyungseopLee: 
class DetectionWSTLoss:
    """
    Detection loss + Attribute classification loss
    attr_weight: weight for attribute losses
    """

    def __init__(self, model, attr_weight=0.1):
        self.det_loss = v8DetectionLoss(model)
        self.attr_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.attr_weight = attr_weight

    def __call__(self, preds, batch):
        det_preds, attr_preds = preds

        # detection loss
        det_loss, det_loss_items = self.det_loss(det_preds, batch)

        # attribute classification loss
        loss_weather   = self.attr_criterion(attr_preds["weather"],   batch["weather"].to(det_loss.device))
        loss_scene     = self.attr_criterion(attr_preds["scene"],     batch["scene"].to(det_loss.device))
        loss_timeofday = self.attr_criterion(attr_preds["timeofday"], batch["timeofday"].to(det_loss.device))

        attr_loss = (loss_weather + loss_timeofday + loss_scene) / 3
        total_loss = det_loss + self.attr_weight * attr_loss

        # @HyungseopLee: loss_items = [box, cls, dfl, weather, scene, timeofday]
        loss_items = torch.cat([
            det_loss_items,
            torch.tensor([
                loss_weather.item(),
                loss_scene.item(),
                loss_timeofday.item(),
            ], device=det_loss.device)
        ])

        return total_loss, loss_items