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


class PolicyNetwork(nn.Module):
    def __init__(self, group_dim: int = 64, path_dim: int = 8, hidden_dim: int = 64,
                 feat: str = "both"):
        super().__init__()
        assert feat in ("input", "pred", "both")
        self.feat = feat
        # Normalise raw GAP vectors (per-channel) before projection -- GAP scales
        # vary widely across layers and unnormalised inputs cripple learning.
        # LazyBatchNorm1d / LazyLinear infer channel counts on first forward.
        if feat in ("input", "both"):
            self.input_proj = nn.Sequential(
                nn.LazyBatchNorm1d(), nn.LazyLinear(group_dim), nn.ReLU(inplace=True))
        if feat in ("pred", "both"):
            self.pred_proj = nn.Sequential(
                nn.LazyBatchNorm1d(), nn.LazyLinear(group_dim), nn.ReLU(inplace=True))
        self.path_embed = nn.Embedding(2, path_dim)  # 0=BASE, 1=SUPER
        n_groups = 2 if feat == "both" else 1
        # [n*d + p] -> [h] -> [1]  (logit / predicted advantage)
        self.head = nn.Sequential(
            nn.Linear(n_groups * group_dim + path_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def logit(self, input_vec: Vec, pred_vec: Vec, path_id: torch.Tensor) -> torch.Tensor:
        pid = self.path_embed(path_id)                 # [B, p]
        parts = []
        if self.feat in ("input", "both"):
            parts.append(self.input_proj(_as_tensor(input_vec)))   # [B, d]
        if self.feat in ("pred", "both"):
            parts.append(self.pred_proj(_as_tensor(pred_vec)))     # [B, d]
        parts.append(pid)
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
