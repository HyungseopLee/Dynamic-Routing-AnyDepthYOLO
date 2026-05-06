# AnyDepth — adaptive router (ported from DynamicDet/models/common.py:AdaptiveRouter)

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sigmoid_ste(x, threshold=0.5):
    """Soft sigmoid with straight-through rounding at `threshold` during training."""
    y_soft = x.sigmoid()
    y_hard = (y_soft > threshold).float()
    return y_hard.detach() - y_soft.detach() + y_soft


class AdaptiveRouter(nn.Module):
    """
    Per-image difficulty scorer for cascaded (SUPER/BASE) deㅋtectors.

    Consumes the essential-path feature (layer 1 output of anydepth-yolov12, P2/4, 128ch)
    and outputs a scalar score in [0, 1]. Higher score => use SUPER; lower => use BASE.
    """

    def __init__(self, in_channels, reduction=4, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        hidden = max(in_channels // reduction, 1)
        self.layer1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.layer2 = nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True)

    def forward(self, x, thres=0.5):
        # print(f"x.shape: {x.shape}") # torch.Size([1, 128, 184, 320])
        x = x.mean(dim=(2, 3), keepdim=True)
        x = self.layer1(x)
        x = F.relu(x, inplace=True)
        x = self.layer2(x).flatten(1)
        # print(f"x.shape: {x.shape}")
        if self.training:
            return _sigmoid_ste(x, threshold=thres)
        return x.sigmoid()
