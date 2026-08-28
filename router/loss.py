"""Router training loss: direct regression of the per-image advantage.

    L = L_acc(R(F), A)

  - A_i = (L_base[i] - L_super[i]).detach() -- per-image advantage of running
          SUPER, a constant w.r.t. the router (the detector is frozen).
  - R(F) is the router's raw scalar output (linear, no sigmoid): a direct
    prediction of A, not a probability.
  - The regression form is ablatable: magnitude losses (mse/mae/huber) vs the
    scale-invariant ranking loss (corr).

"""

import torch
import torch.nn.functional as F


class RouterLoss:
    def __init__(self, regress_loss: str = "mse"):
        """
        Args:
            regress_loss: mse | mae | huber | corr
        """
        self.regress_loss = regress_loss

    def __call__(self, pred: torch.Tensor,
                 l_base: torch.Tensor, l_super: torch.Tensor):
        """
        Args:
            pred:    [B] or [B,1] router output -- predicted advantage (requires grad).
            l_base:  [B] per-image detection loss under BASE  (no grad).
            l_super: [B] per-image detection loss under SUPER (no grad).
        Returns:
            (total_loss, dict of components)
        """
        adv_raw = (l_base.view(-1) - l_super.view(-1)).detach()  # A_i, constant
        pred = pred.view(-1)

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

        return l_acc, {
            "l_acc": float(l_acc.detach()),
            "p_super_mean": float(pred.mean().detach()),
            "adv_mean_raw": float(adv_raw.mean()),
            "total": float(l_acc.detach()),
        }
