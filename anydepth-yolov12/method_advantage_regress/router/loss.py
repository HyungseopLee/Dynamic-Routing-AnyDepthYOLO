"""Router training loss for method01 (2-level, path B: deterministic advantage).

    L = L_acc + lambda_flops * L_flops + lambda_uni * L_uni

  - L_acc   = - mean_i ( p_super_i * A_i )
              A_i = (L_base[i] - L_super[i]).detach()  -- per-image advantage of
              running SUPER, a constant w.r.t. the router (detector is frozen).
              p_super_i in (0,1) is the router's soft probability of SUPER.
              Minimising raises p_super where SUPER actually helps (A>0) and
              lowers it where it does not (A<=0). No Gumbel/sampling needed: the
              objective is linear in the decision so the soft prob equals the
              expected hard-gate loss exactly (zero variance).
  - L_flops = expected GFLOPs of the (soft) choice, normalised by SUPER cost,
              so it lives in [g_base/g_super, 1]. Constants from the offline table.
  - L_uni   = (mean_i p_super_i - uni_target)^2  -- weak anti-collapse (small
              lambda_uni); the real compute/AP trade-off is set by lambda_flops.

Optionally standardises A across the batch (z-score) so no single high-loss
image dominates the gradient and lambda_flops stays interpretable.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


class RouterLoss:
    def __init__(self, flops_table_path: str,
                 lambda_flops: float = 1.0, lambda_uni: float = 0.1,
                 uni_target: float = 0.5, standardize_adv: bool = True,
                 mode: str = "advantage", margin: float = 0.0,
                 regress_loss: str = "mse"):
        """
        mode:
          "advantage" -> L_acc = -mean(p * A)      (A optionally standardised)
          "bce"       -> L_acc = BCE(p, 1[A>0]) over samples with |A| >= margin
                         (ambiguous |A|<margin images are masked out of L_acc;
                          flops then sends them to BASE). target uses the SIGN of
                          the advantage, so small noise magnitude no longer drives
                          the gradient.
        margin: advantage band (in raw loss units) excluded from BCE.
        """
        self.lambda_flops = lambda_flops
        self.lambda_uni = lambda_uni
        self.uni_target = uni_target
        self.standardize_adv = standardize_adv
        self.mode = mode
        self.margin = margin
        self.regress_loss = regress_loss

        table = json.loads(Path(flops_table_path).read_text())
        self.g_base = float(table["actions"]["0_base"]["gflops"])
        self.g_super = float(table["actions"]["1_super"]["gflops"])

    def __call__(self, p_super: torch.Tensor,
                 l_base: torch.Tensor, l_super: torch.Tensor):
        """
        Args:
            p_super: [B] or [B,1] soft prob of SUPER (router output, requires grad).
            l_base:  [B] per-image detection loss under BASE  (no grad).
            l_super: [B] per-image detection loss under SUPER (no grad).
        Returns:
            (total_loss, dict of components)
        """
        adv_raw = (l_base.view(-1) - l_super.view(-1)).detach()  # A_i, constant

        if self.mode == "regress":
            # router output is a direct advantage prediction (linear, no sigmoid).
            # The regression loss form is ablatable (C1): magnitude losses
            # (mse/mae/huber) vs scale-invariant ranking (corr). Trade-off handled
            # by the eval threshold, so no flops/uni needed.
            pred = p_super.view(-1)
            rl = self.regress_loss
            if rl == "mse":
                l_acc = F.mse_loss(pred, adv_raw)
            elif rl == "mae":
                l_acc = F.l1_loss(pred, adv_raw)
            elif rl == "huber":
                l_acc = F.huber_loss(pred, adv_raw, delta=0.1)
            elif rl == "corr":
                # 1 - Pearson(pred, A) over the batch (scale-invariant ranking)
                a = pred - pred.mean(); b = adv_raw - adv_raw.mean()
                l_acc = 1.0 - (a * b).sum() / (a.norm() * b.norm() + 1e-8)
            else:
                raise ValueError(f"unknown regress_loss: {rl}")
            total = l_acc
            return total, {
                "l_acc": float(l_acc.detach()),
                "l_flops": 0.0, "l_uni": 0.0,
                "p_super_mean": float(pred.mean().detach()),
                "adv_mean_raw": float(adv_raw.mean()),
                "valid_frac": 1.0,
                "total": float(total.detach()),
            }

        p = p_super.view(-1).clamp(1e-6, 1 - 1e-6)

        if self.mode == "bce":
            valid = adv_raw.abs() >= self.margin           # drop ambiguous band
            target = (adv_raw > 0).float()
            if valid.any():
                l_acc = F.binary_cross_entropy(p[valid], target[valid])
            else:
                l_acc = torch.zeros((), device=p.device)
            valid_frac = valid.float().mean()
        else:
            adv = adv_raw
            if self.standardize_adv and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            l_acc = -(p * adv).mean()
            valid_frac = torch.ones((), device=p.device)

        flops = (1 - p) * self.g_base + p * self.g_super       # [B]
        l_flops = flops.mean() / self.g_super

        l_uni = (p.mean() - self.uni_target) ** 2

        total = l_acc + self.lambda_flops * l_flops + self.lambda_uni * l_uni
        return total, {
            "l_acc": float(l_acc.detach()),
            "l_flops": float(l_flops.detach()),
            "l_uni": float(l_uni.detach()),
            "p_super_mean": float(p.mean().detach()),
            "adv_mean_raw": float(adv_raw.mean()),
            "valid_frac": float(valid_frac),
            "total": float(total.detach()),
        }
