"""Diagnose the router ceiling on BDD: is low val_corr a *capacity/training*
problem or an *information* problem (A not predictable from the cached features)?

Trains progressively stronger regressors of A = loss_base - loss_super on the
cached GAP features (train cache) and reports val Pearson corr(A_hat, A):
  linear / small-MLP / big-MLP / gradient-boosting.
If even GBT on the full GAP vector can't beat ~0.2, the bottleneck is the
feature/target (information), not the model -> we must change the action space
or the target, not the optimizer. Also reports the loss-space ORACLE routing
ceiling (route super iff A>0) so we know the max attainable gain.
"""
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

O = "method02_advantage_regress_tinyConv/outputs/bdd100k"
tr = torch.load(f"{O}/cache_train_both.pt", map_location="cpu", weights_only=False)
va = torch.load(f"{O}/cache_val_both.pt", map_location="cpu", weights_only=False)


def feats(c):
    # GAP the 2x2 grid -> global vector, concat backbone+neck (path_id=BASE)
    i = c["input_base"].mean(dim=(2, 3)); p = c["pred_base"].mean(dim=(2, 3))
    return torch.cat([i, p], 1).numpy()


Xtr, Xva = feats(tr), feats(va)
Atr = (tr["loss_base"] - tr["loss_super"]).view(-1).numpy()
Ava = (va["loss_base"] - va["loss_super"]).view(-1).numpy()
print(f"train {Xtr.shape} val {Xva.shape}  A_va std={Ava.std():.4f}")


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# standardise
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd

# 1) linear (Ridge)
for al in (1.0, 10.0, 100.0):
    r = Ridge(alpha=al).fit(Xtr_n, Atr)
    print(f"ridge a={al:6.1f}  val_corr={corr(r.predict(Xva_n), Ava):.3f}")

# 2) torch MLPs of increasing capacity
import torch.nn as nn
Xt = torch.tensor(Xtr_n, dtype=torch.float32); At = torch.tensor(Atr, dtype=torch.float32)
Xv = torch.tensor(Xva_n, dtype=torch.float32)
for hid in (64, 256, 1024):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(Xt.shape[1], hid), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    best = -1
    for ep in range(120):
        net.train(); perm = torch.randperm(len(Xt))
        for j in range(0, len(Xt), 512):
            b = perm[j:j + 512]; opt.zero_grad()
            loss = ((net(Xt[b]).squeeze(1) - At[b]) ** 2).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            c = corr(net(Xv).squeeze(1).numpy(), Ava)
        best = max(best, c)
    print(f"mlp hid={hid:4d}  best val_corr={best:.3f}")

# 3) gradient boosting (nonlinear, low-variance)
g = HistGradientBoostingRegressor(max_depth=4, max_iter=400, learning_rate=0.05,
                                  l2_regularization=1.0).fit(Xtr, Atr)
print(f"GBT          val_corr={corr(g.predict(Xva), Ava):.3f}")

# 4) loss-space ORACLE ceiling: route super iff A>0 (and at budgets)
print("\n--- loss-space oracle routing (val) ---")
lb = va["loss_base"].view(-1).numpy(); ls = va["loss_super"].view(-1).numpy()
print(f"always_base mean loss = {lb.mean():.4f}")
print(f"always_super          = {ls.mean():.4f}  (gain {lb.mean()-ls.mean():.4f})")
print(f"oracle (min per img)  = {np.minimum(lb, ls).mean():.4f}  "
      f"(gain {lb.mean()-np.minimum(lb,ls).mean():.4f})")
for b in (20, 50, 80):
    k = int(b / 100 * len(Ava)); thr = np.sort(Ava)[::-1][k - 1]
    sel = Ava >= thr
    routed = np.where(sel, ls, lb).mean()
    print(f"  oracle @ {b}% super: loss={routed:.4f}  (gain {lb.mean()-routed:.4f})")
