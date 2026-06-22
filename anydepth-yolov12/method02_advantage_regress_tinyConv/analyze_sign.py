"""Sign-agreement analysis for the advantage router on BDD100K.

Question: in frames where BASE is actually better (true advantage A = L_base - L_super
< 0, i.e. SUPER *hurts*), does the router predict a negative A-hat and route to BASE?

Reports: sign confusion matrix, recall on the A<0 (base-better) subset, |A|-weighted
sign error, and the accuracy recovered vs an always-super policy. Also dumps the
worst false-positives (A<0 but router says SUPER) for qualitative inspection.

    python -m method02_advantage_regress_tinyConv.analyze_sign \
        --cache outputs/bdd100k/cache_val_g2.pt \
        --policy outputs/bdd100k/policy_scenario_s0.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork


@torch.no_grad()
def predict(cache, policy, device):
    c = torch.load(cache, map_location="cpu", weights_only=False)
    lb, ls = c["loss_base"].view(-1), c["loss_super"].view(-1)
    ckpt = torch.load(policy, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 64), feat=a.get("feat", "input"),
                        norm=a.get("norm", "batch"), dropout=0.0).to(device).eval()
    inp = c["input_base"].to(device)
    pid = torch.zeros(inp.shape[0], dtype=torch.long, device=device)
    net(inp[:2], None, pid[:2])              # materialise LazyLinear
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    ahat = net.logit(inp, None, pid).view(-1).cpu()
    return c["im_file"], lb, ls, ahat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/bdd100k/cache_val_g2.pt")
    ap.add_argument("--policy", default="outputs/bdd100k/policy_scenario_s0.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump", default="outputs/bdd100k/sign_analysis.csv")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    files, lb, ls, ahat = predict(args.cache, args.policy, dev)
    A = (lb - ls).numpy(); H = ahat.numpy()
    N = len(A)
    pos, neg = A > 0, A < 0      # super-better / base-better

    print(f"N={N}   super-better(A>0)={pos.mean()*100:.1f}%   base-better(A<0)={neg.mean()*100:.1f}%")
    # sign confusion (router decision = sign of A-hat at tau=0)
    pred_super = H > 0
    tp = (pred_super & pos).sum(); fn = (~pred_super & pos).sum()
    fp = (pred_super & neg).sum(); tn = (~pred_super & neg).sum()
    print("\n== sign confusion @ tau=0 (router_super vs truth) ==")
    print(f"            true A>0   true A<0")
    print(f"pred SUPER   {tp:6d}     {fp:6d}")
    print(f"pred BASE    {fn:6d}     {tn:6d}")
    print(f"\noverall sign accuracy : {(tp+tn)/N*100:.1f}%")
    print(f"A<0 recall (base kept): {tn/(tn+fp)*100:.1f}%   <-- key: base-better frames sent to BASE")
    print(f"A>0 recall (super used): {tp/(tp+fn)*100:.1f}%")

    # |A|-weighted sign error: misses weighted by how much they cost/help
    w = np.abs(A)
    wrong = (pred_super != pos)
    print(f"\n|A|-weighted sign error: {w[wrong].sum()/w.sum()*100:.1f}% of total |advantage| misrouted")

    # accuracy recovered vs always-super (loss terms; lower is better)
    L_super = ls.numpy().mean()
    L_router = np.where(pred_super, ls.numpy(), lb.numpy()).mean()
    L_oracle = np.minimum(lb.numpy(), ls.numpy()).mean()
    print(f"\nmean loss  always-super={L_super:.4f}  router={L_router:.4f}  oracle={L_oracle:.4f}")
    print(f"router recovers {(L_super-L_router)/(L_super-L_oracle)*100:.1f}% of the "
          f"always-super->oracle accuracy gap")
    print(f"router SUPER usage: {pred_super.mean()*100:.1f}%")

    # rank quality on the base-better subset: does A-hat rank A<0 frames low?
    from scipy.stats import spearmanr
    rho = spearmanr(A, H).correlation
    print(f"\nSpearman rho(A, A-hat) = {rho:.3f}  (rank agreement over full range)")

    # dump worst false positives (base-better but router most confident on SUPER)
    fp_idx = np.where(neg)[0]
    fp_idx = fp_idx[np.argsort(-H[fp_idx])][:30]
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dump, "w") as f:
        f.write("im_file,A_true,A_hat,loss_base,loss_super\n")
        for i in np.argsort(A):     # full table, ascending A (worst base-better first)
            f.write(f"{files[i]},{A[i]:.4f},{H[i]:.4f},{lb[i]:.4f},{ls[i]:.4f}\n")
    print(f"\n[*] full per-image table -> {args.dump}")
    print("    (top base-better frames the router wrongly sent to SUPER:)")
    for i in fp_idx[:8]:
        print(f"      A={A[i]:+.3f}  A-hat={H[i]:+.3f}  {Path(files[i]).name}")


if __name__ == "__main__":
    main()
