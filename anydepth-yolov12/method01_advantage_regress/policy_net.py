"""Policy network for the 2-level depth decision (method01, path B: deterministic).

Emits a single scalar per sample:
    p_super = sigmoid(logit) = P(action == SUPER)   (0=BASE essential, 1=SUPER full)

No Gumbel sampling / straight-through: because the detector is frozen and the
per-image advantage A = (L_base - L_super) is a constant, the policy objective
is linear in the decision, so the soft probability p_super equals the *expected*
hard-gate loss exactly (zero variance). Training is plain backprop through the
sigmoid; eval thresholds p_super (>= thres -> SUPER), which lets one trained
model trace the whole AP-vs-FLOPs curve via the threshold knob.

Inputs are the pre-concatenated context vectors (built by feature_tap /
the caching script):
    input_vec : concat(GAP of backbone layers 4,6,8)      e.g. [B, 768]
    pred_vec  : concat(GAP of neck layers 14,17,20)        e.g. [B, 640]
    path_id   : which path produced these feats (0=BASE,1=SUPER)

    input_vec --LazyLinear->ReLU--> [B,d]  ┐
    pred_vec  --LazyLinear->ReLU--> [B,d]  ├ concat [B,2d+p] --FC->ReLU->FC--> logit [B,1]
    path_id   --Embedding(2,p)----> [B,p]  ┘
"""

from typing import List, Union

import torch
import torch.nn as nn

Vec = Union[torch.Tensor, List[torch.Tensor]]


def _as_tensor(v: Vec) -> torch.Tensor:
    """Accept either a pre-concatenated [B,C] tensor or a list of [B,Ci] tensors."""
    return v if isinstance(v, torch.Tensor) else torch.cat(v, dim=1)


class LazyLayerNorm(nn.Module):
    """LayerNorm whose normalized_shape is inferred on the first forward.

    Unlike BatchNorm, this normalises each sample over its own channels, so it is
    independent of batch size / other frames -- a natural fit for batch-1 video
    inference. Built lazily to mirror LazyBatchNorm1d (no channel count hardcoded).
    """

    def __init__(self):
        super().__init__()
        self.ln = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ln is None:
            self.ln = nn.LayerNorm(x.shape[1]).to(x.device, x.dtype)
        return self.ln(x)


def _norm(kind: str) -> nn.Module:
    if kind == "batch":
        return nn.LazyBatchNorm1d()
    if kind == "layer":
        return LazyLayerNorm()
    if kind == "none":
        return nn.Identity()  # feed raw GAP straight into the projection
    raise ValueError(f"unknown norm: {kind}")


class PolicyNetwork(nn.Module):
    def __init__(self, group_dim: int = 64, path_dim: int = 8, hidden_dim: int = 64,
                 feat: str = "both", norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        assert feat in ("input", "pred", "both")
        self.feat = feat
        self.norm = norm
        self.dropout = dropout
        # Normalise raw GAP vectors before projection -- GAP scales vary widely
        # across layers and unnormalised inputs cripple learning. norm="batch"
        # uses per-channel BatchNorm (batch stats); norm="layer" normalises each
        # sample over its channels (batch-size independent). LazyLinear infers
        # channel counts on first forward.
        if feat in ("input", "both"):
            self.input_proj = nn.Sequential(
                _norm(norm), nn.LazyLinear(group_dim), nn.ReLU(inplace=True))
        if feat in ("pred", "both"):
            self.pred_proj = nn.Sequential(
                _norm(norm), nn.LazyLinear(group_dim), nn.ReLU(inplace=True))
        # prev-action (path) embedding. path_dim=0 -> no embedding (policy ignores
        # which path produced the features); kept out of the graph so the head input
        # shrinks accordingly.
        self.path_dim = path_dim
        self.path_embed = nn.Embedding(2, path_dim) if path_dim > 0 else None  # 0=BASE,1=SUPER
        n_groups = 2 if feat == "both" else 1
        # [n*d + p] -> [h] -> [1]  (logit / predicted advantage). Dropout module is
        # only inserted when dropout>0 so old (no-dropout) checkpoints stay loadable.
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
        """Training forward: returns p_super [B,1] = sigmoid(logit). Differentiable."""
        return torch.sigmoid(self.logit(input_vec, pred_vec, path_id))

    @torch.no_grad()
    def score(self, input_vec: Vec, pred_vec: Vec, path_id: torch.Tensor) -> torch.Tensor:
        """Eval-time P(super) score [B,1]. Threshold downstream."""
        return torch.sigmoid(self.logit(input_vec, pred_vec, path_id))

    @torch.no_grad()
    def act(self, input_vec: Vec, pred_vec: Vec, path_id: torch.Tensor,
            thres: float = 0.5) -> torch.Tensor:
        """Eval-time decision: score >= thres -> SUPER. Returns action [B] in {0,1}."""
        s = self.score(input_vec, pred_vec, path_id).view(-1)
        return (s >= thres).long()
