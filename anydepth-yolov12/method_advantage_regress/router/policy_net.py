"""Router policy networks for 2-level depth routing.

Two architectures:

PolicyNetwork (TinyConv):
    Keeps spatial structure of pooled feature grids [B, C, G, G].
    Per group: BN2d -> 1x1 conv -> ReLU -> 3x3 depthwise -> ReLU -> GAP -> [B, d]
    Concat groups + path embed -> FC -> ReLU -> FC -> logit [B, 1].

GapMlpNet (GAP-MLP):
    Operates on spatially pre-averaged features [B, C] (G=1 equivalent).
    Per group: BN1d -> Linear(C->d) -> ReLU -> [B, d]
    Concat groups + path embed -> FC -> ReLU -> FC -> logit [B, 1].
    Callers must pool 4D grids to [B, C] before passing to this network.
"""

from typing import List, Union

import torch
import torch.nn as nn

Vec = Union[torch.Tensor, List[torch.Tensor]]


def _as_tensor(v: Vec) -> torch.Tensor:
    """Accept a pre-concatenated [B,C,G,G] tensor or a list of [B,Ci,G,G] grids
    (concatenated along the channel dim)."""
    return v if isinstance(v, torch.Tensor) else torch.cat(v, dim=1)


def _norm2d(kind: str) -> nn.Module:
    if kind == "batch":
        return nn.LazyBatchNorm2d()
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind}")


class ConvGroupProj(nn.Module):
    """Tiny spatial projection of one feature group: BN2d -> 1x1 (C->d) -> ReLU
    -> 3x3 depthwise (d->d) -> ReLU -> global average pool -> [B, d].

    Cheap: 1x1 is C*d*G*G MACs, depthwise is d*9*G*G. For C=768,d=64,G=4 that is
    ~0.8M MACs/group -- still <0.01% of the detector, while letting the router use
    coarse spatial layout (where in the frame the hard content is) rather than a
    single global average."""

    def __init__(self, group_dim: int, norm: str):
        super().__init__()
        self.norm = _norm2d(norm)
        self.point = nn.LazyConv2d(group_dim, kernel_size=1)
        self.depth = nn.Conv2d(group_dim, group_dim, kernel_size=3, padding=1, groups=group_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.point(self.norm(x)))
        x = self.act(self.depth(x))
        return x.mean(dim=(2, 3))   # GAP -> [B, d]


class PolicyNetwork(nn.Module):
    def __init__(self, group_dim: int = 64, path_dim: int = 8, hidden_dim: int = 64,
                 feat: str = "both", norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        assert feat in ("input", "pred", "both")
        self.feat = feat
        self.norm = norm
        self.dropout = dropout
        if feat in ("input", "both"):
            self.input_proj = ConvGroupProj(group_dim, norm)
        if feat in ("pred", "both"):
            self.pred_proj = ConvGroupProj(group_dim, norm)
        self.path_dim = path_dim
        self.path_embed = nn.Embedding(2, path_dim) if path_dim > 0 else None
        n_groups = 2 if feat == "both" else 1
        layers = [nn.Linear(n_groups * group_dim + path_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(*layers)

    def logit(self, input_vec: Vec, pred_vec: Vec, path_id: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.feat in ("input", "both"):
            parts.append(self.input_proj(_as_tensor(input_vec)))   # [B, d]
        if self.feat in ("pred", "both"):
            parts.append(self.pred_proj(_as_tensor(pred_vec)))     # [B, d]
        if self.path_embed is not None:
            parts.append(self.path_embed(path_id))                 # [B, p]
        return self.head(torch.cat(parts, dim=1))      # [B, 1]

    def forward(self, input_vec: Vec, pred_vec: Vec, path_id: torch.Tensor) -> torch.Tensor:
        """Training forward: returns p_super [B,1] = sigmoid(logit)."""
        return torch.sigmoid(self.logit(input_vec, pred_vec, path_id))

    @torch.no_grad()
    def score(self, input_vec: Vec, pred_vec: Vec, path_id: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(input_vec, pred_vec, path_id))

    @torch.no_grad()
    def act(self, input_vec: Vec, pred_vec: Vec, path_id: torch.Tensor,
            thres: float = 0.5) -> torch.Tensor:
        s = self.score(input_vec, pred_vec, path_id).view(-1)
        return (s >= thres).long()


class _LinearGroupProj(nn.Module):
    """1D group projection for GAP-MLP: BN1d -> LazyLinear(C->d) -> ReLU."""

    def __init__(self, group_dim: int, norm: str):
        super().__init__()
        if norm == "batch":
            self.norm = nn.LazyBatchNorm1d()
        else:
            self.norm = nn.Identity()
        self.fc = nn.LazyLinear(group_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(self.norm(x)))


class GapMlpNet(nn.Module):
    """GAP-MLP router: expects spatially pre-averaged features [B, C].

    Same interface as PolicyNetwork but all weights are 2D (Linear, not Conv),
    so checkpoint detection (checking for 4D weight tensors) correctly
    distinguishes this from the TinyConv router.

    Usage:
        net = GapMlpNet(feat="both")
        # pool 4D grids before calling:
        inp = cache["input_base"].mean(dim=(2, 3))  # [B, C]
        pred = cache["pred_base"].mean(dim=(2, 3))
        logit = net.logit(inp, pred, path_id)
    """

    def __init__(self, group_dim: int = 64, path_dim: int = 8, hidden_dim: int = 64,
                 feat: str = "both", norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        assert feat in ("input", "pred", "both")
        self.feat = feat
        if feat in ("input", "both"):
            self.input_proj = _LinearGroupProj(group_dim, norm)
        if feat in ("pred", "both"):
            self.pred_proj = _LinearGroupProj(group_dim, norm)
        self.path_dim = path_dim
        self.path_embed = nn.Embedding(2, path_dim) if path_dim > 0 else None
        n_groups = 2 if feat == "both" else 1
        layers = [nn.Linear(n_groups * group_dim + path_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(*layers)

    def logit(self, input_vec: torch.Tensor, pred_vec: torch.Tensor,
              path_id: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.feat in ("input", "both"):
            parts.append(self.input_proj(input_vec))
        if self.feat in ("pred", "both"):
            parts.append(self.pred_proj(pred_vec))
        if self.path_embed is not None:
            parts.append(self.path_embed(path_id))
        return self.head(torch.cat(parts, dim=1))

    def forward(self, input_vec: torch.Tensor, pred_vec: torch.Tensor,
                path_id: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(input_vec, pred_vec, path_id))

    @torch.no_grad()
    def score(self, input_vec: torch.Tensor, pred_vec: torch.Tensor,
              path_id: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(input_vec, pred_vec, path_id))

    @torch.no_grad()
    def act(self, input_vec: torch.Tensor, pred_vec: torch.Tensor,
            path_id: torch.Tensor, thres: float = 0.5) -> torch.Tensor:
        s = self.score(input_vec, pred_vec, path_id).view(-1)
        return (s >= thres).long()
