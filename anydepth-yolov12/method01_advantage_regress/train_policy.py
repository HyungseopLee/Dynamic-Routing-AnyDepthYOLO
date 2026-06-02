"""Train the 2-level depth policy on cached features (method01, path B).

Reads the offline cache (build_cache.py) so no detector forward happens here --
each epoch is a cheap MLP pass over precomputed vectors.

Per step (image i, batch B):
  - sample prev_action ~ Uniform{0,1}  (simulates the path the previous frame
    used at eval time; selects which cached feature vector feeds the policy and
    sets path_id).
  - p_super = policy(input_vec[prev], pred_vec[prev], path_id=prev)
  - L = -mean(p_super * A) + lambda_flops*L_flops + lambda_uni*L_uni
    with A = (loss_base - loss_super).detach()
  - update policy only.

Usage:
    # kitti  (--dataset scopes all I/O under outputs/kitti/)
    python method01_advantage_regress/train_policy.py \
        --dataset kitti --epochs 50 \
        --lambda_flops 1.0 --lambda_uni 0.1 --mode regress --tag regress

    # bdd100k  (reads outputs/bdd100k/cache_*.pt + flops_table.json from build_*)
    # image size is independent here: the cache stores GAP feature vectors and
    # per-image losses, so the policy MLP is identical to kitti regardless of
    # the 720x1280 vs 384x1248 input. Only the flops_table.json values differ.
    python method01_advantage_regress/train_policy.py \
        --dataset bdd100k --epochs 50 \
        --lambda_flops 1.0 --lambda_uni 0.1 --mode regress --tag regress
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from method01_advantage_regress.loss import PolicyLoss
from method01_advantage_regress.policy_net import PolicyNetwork


def load_cache(path, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "input_base": c["input_base"].to(device),
        "pred_base": c["pred_base"].to(device),
        "input_super": c["input_super"].to(device),
        "pred_super": c["pred_super"].to(device),
        "loss_base": c["loss_base"].to(device),
        "loss_super": c["loss_super"].to(device),
    }


def make_loader(cache, batch, shuffle):
    idx = torch.arange(cache["loss_base"].shape[0])
    return DataLoader(TensorDataset(idx), batch_size=batch, shuffle=shuffle)


def gather(cache, idx, prev_action):
    """Select per-sample feature vectors by prev_action path (0=base,1=super)."""
    use_super = prev_action.bool()
    inp = torch.where(use_super[:, None], cache["input_super"][idx], cache["input_base"][idx])
    prd = torch.where(use_super[:, None], cache["pred_super"][idx], cache["pred_base"][idx])
    return inp, prd


@torch.no_grad()
def val_corr(net, cache):
    """Pearson corr between predicted A-hat and true advantage A on the val set,
    using prev_action=BASE (path_id=0) -- matches how eval thresholds A-hat. This
    is scale-invariant, so it tracks routing/ranking quality rather than MSE."""
    net.eval()
    A = (cache["loss_base"].view(-1) - cache["loss_super"].view(-1))
    pid = torch.zeros(A.shape[0], dtype=torch.long, device=A.device)
    ah = net.logit(cache["input_base"], cache["pred_base"], pid).view(-1)
    a = ah - ah.mean(); b = A - A.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def run_epoch(net, cache, loader, lossfn, opt=None):
    train = opt is not None
    net.train(train)
    agg = {}
    n = 0
    for (idx,) in loader:
        idx = idx.to(cache["loss_base"].device)
        B = idx.shape[0]
        prev = torch.randint(0, 2, (B,), device=idx.device)
        inp, prd = gather(cache, idx, prev)
        lb, ls = cache["loss_base"][idx], cache["loss_super"][idx]

        with torch.set_grad_enabled(train):
            # regress mode predicts advantage directly (linear logit); others use prob
            out = net.logit(inp, prd, prev) if lossfn.mode == "regress" else net(inp, prd, prev)
            total, comp = lossfn(out, lb, ls)
        if train:
            opt.zero_grad(); total.backward(); opt.step()
        for k, v in comp.items():
            agg[k] = agg.get(k, 0.0) + v * B
        n += B
    return {k: v / n for k, v in agg.items()}


OUT = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti", help="output scope: outputs/<dataset>/")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--val_cache", default=None)
    ap.add_argument("--flops", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_flops", type=float, default=1.0)
    ap.add_argument("--lambda_uni", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--group_dim", type=int, default=64)
    ap.add_argument("--path_dim", type=int, default=8,
                    help="prev-action embedding dim (0 = no path embedding)")
    ap.add_argument("--feat", default="both", choices=["input", "pred", "both"],
                    help="which context feature(s) the policy uses")
    ap.add_argument("--norm", default="batch", choices=["batch", "layer", "none"],
                    help="GAP normalisation before projection (batch=BatchNorm1d, layer=LayerNorm, none=Identity)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout in the head (regularisation)")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay (L2 regularisation)")
    ap.add_argument("--select", default="val_mse", choices=["val_mse", "val_corr", "last"],
                    help="checkpoint selection: val_mse=min val MSE(A-hat,A); "
                         "val_corr=max val corr(A-hat,A) (ranking, matches eval thresholding); "
                         "last=final epoch (no early stop)")
    ap.add_argument("--mode", default="advantage", choices=["advantage", "bce", "regress"])
    ap.add_argument("--margin", type=float, default=0.0,
                    help="bce mode: exclude |A|<margin images from L_acc")
    ap.add_argument("--seed", type=int, default=0, help="random seed (weight init + prev_action sampling)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--logdir", default=None, help="directory for per-run training logs")
    ap.add_argument("--tag", default="", help="optional run name suffix for the log file")
    args = ap.parse_args()
    base = OUT / args.dataset
    if args.cache is None:     args.cache = str(base / "cache_train.pt")
    if args.val_cache is None: args.val_cache = str(base / "cache_val.pt")
    if args.flops is None:     args.flops = str(base / "flops_table.json")
    if args.out is None:       args.out = str(base / "policy.pt")
    if args.logdir is None:    args.logdir = str(base / "logs")

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    train_cache = load_cache(args.cache, device)
    val_cache = load_cache(args.val_cache, device) if Path(args.val_cache).exists() else None

    net = PolicyNetwork(group_dim=args.group_dim, path_dim=args.path_dim, hidden_dim=args.hidden,
                        feat=args.feat, norm=args.norm, dropout=args.dropout).to(device)
    # materialise LazyLinear shapes with a dummy forward
    with torch.no_grad():
        net(train_cache["input_base"][:2], train_cache["pred_base"][:2],
            torch.zeros(2, dtype=torch.long, device=device))

    lossfn = PolicyLoss(args.flops, lambda_flops=args.lambda_flops, lambda_uni=args.lambda_uni,
                        mode=args.mode, margin=args.margin)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(train_cache, args.batch, shuffle=True)
    val_loader = make_loader(val_cache, args.batch, shuffle=False) if val_cache else None

    # per-run log file in method01/logs
    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_lf{args.lambda_flops}_lu{args.lambda_uni}" + (f"_{args.tag}" if args.tag else "")
    logpath = logdir / f"{name}.log"
    history = []
    with open(logpath, "w") as f:
        f.write(f"# {json.dumps(vars(args))}\n")
        f.write("epoch\ttotal\tl_acc\tl_flops\tl_uni\tp_super\tval_p_super\tval_l_acc\n")

    # track the val-best checkpoint (lowest val l_acc); falls back to train l_acc
    # if no val cache. Saving the best -- not the last epoch -- guards against any
    # late-epoch overfitting on the held-out images.
    import copy
    best_obj = float("inf")          # objective to MINIMISE (sign-flipped where needed)
    best_metric = float("nan")        # human-readable value of the selection metric
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = -1
    sel = args.select if val_loader or args.select == "last" else "train_l_acc"

    for ep in range(args.epochs):
        tr = run_epoch(net, train_cache, train_loader, lossfn, opt)
        msg = (f"ep {ep:3d} | train total {tr['total']:.4f} "
               f"acc {tr['l_acc']:.4f} flops {tr['l_flops']:.4f} "
               f"p_super {tr['p_super_mean']:.3f}")
        va = run_epoch(net, val_cache, val_loader, lossfn, None) if val_loader else {}
        if va:
            msg += f" || val p_super {va['p_super_mean']:.3f} acc {va['l_acc']:.4f}"
        # selection objective (lower=better)
        if args.select == "last":
            obj, metric = -ep, float(ep)
        elif args.select == "val_corr" and val_loader:
            c = val_corr(net, val_cache); obj, metric = -c, c
            msg += f" corr {c:.3f}"
        else:  # val_mse (or no val -> train mse)
            metric = va["l_acc"] if val_loader else tr["l_acc"]
            obj = metric
        if obj < best_obj:
            best_obj = obj; best_metric = metric; best_epoch = ep
            best_state = copy.deepcopy(net.state_dict())
            msg += "  *best*"
        print(msg)
        history.append({"epoch": ep, **{f"tr_{k}": v for k, v in tr.items()},
                        **{f"va_{k}": v for k, v in va.items()}})
        with open(logpath, "a") as f:
            f.write(f"{ep}\t{tr['total']:.5f}\t{tr['l_acc']:.5f}\t{tr['l_flops']:.5f}\t"
                    f"{tr['l_uni']:.5f}\t{tr['p_super_mean']:.4f}\t"
                    f"{va.get('p_super_mean', float('nan')):.4f}\t{va.get('l_acc', float('nan')):.5f}\n")
    (logdir / f"{name}.json").write_text(json.dumps(history, indent=2))
    print(f"[*] log -> {logpath}")
    print(f"[*] best {sel}={best_metric:.5f} @ epoch {best_epoch} (of {args.epochs})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state,
                "args": vars(args),
                "best_epoch": best_epoch, "best_metric": best_metric, "select_on": sel}, out)
    print(f"[*] saved policy (val-best) -> {out}")


if __name__ == "__main__":
    main()
