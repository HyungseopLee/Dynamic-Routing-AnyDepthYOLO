"""Train the depth router on cached features.

Reads the offline cache (train/build_cache.py) so no detector forward happens
here -- each epoch is a cheap MLP/TinyConv pass over precomputed vectors.

Per step (image i, batch B):
  - sample prev_action ~ Uniform{0,1}  (simulates the path the previous frame
    used at eval time; selects which cached feature vector feeds the policy and
    sets path_id).
  - p_super = router(input_vec[prev], pred_vec[prev], path_id=prev)
  - L = -mean(p_super * A) + lambda_flops*L_flops + lambda_uni*L_uni
    with A = (loss_base - loss_super).detach()
  - update router only.

Usage (run from repo root):
    # KITTI  (reads outputs/kitti/cache_{train,val}.pt)
    python -m method_advantage_regress.train.train_policy \
        --dataset kitti --feat both --norm batch \
        --cache method_advantage_regress/outputs/kitti/cache_train_both.pt \
        --val_cache method_advantage_regress/outputs/kitti/cache_val_both.pt \
        --epochs 50 --select val_corr --mode regress --seed 0

    # BDD100K  (reads outputs/bdd100k/cache_{train,val}_both.pt)
    python -m method_advantage_regress.train.train_policy \
        --dataset bdd100k --feat both --norm batch \
        --cache method_advantage_regress/outputs/bdd100k/cache_train_both.pt \
        --val_cache method_advantage_regress/outputs/bdd100k/cache_val_both.pt \
        --epochs 30 --select val_corr --mode regress --seed 0 \
        --out method_advantage_regress/outputs/bdd100k/policy_s0.pt

    # Waymo  (reads outputs/waymo/cache_{train,val}_both.pt; fp16 cache)
    python -m method_advantage_regress.train.train_policy \
        --dataset waymo --feat both --norm batch \
        --cache method_advantage_regress/outputs/waymo/cache_train_both.pt \
        --val_cache method_advantage_regress/outputs/waymo/cache_val_both.pt \
        --epochs 50 --select val_corr --mode regress --seed 0 \
        --out method_advantage_regress/outputs/waymo/policy_both_0.pt

    # GAP-MLP router (--arch gapmpl; same cache, spatial dims are pooled internally)
    python -m method_advantage_regress.train.train_policy \
        --dataset bdd100k --arch gapmpl --feat both --norm batch \
        --cache method_advantage_regress/outputs/bdd100k/cache_train_both.pt \
        --val_cache method_advantage_regress/outputs/bdd100k/cache_val_both.pt \
        --epochs 30 --select val_corr --mode regress --seed 0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from method_advantage_regress.router.loss import PolicyLoss
from method_advantage_regress.router.policy_net import GapMlpNet, PolicyNetwork
from method_advantage_regress.router.feature_tap import INPUT_LEVEL_LAYERS


def slice_levels(cache, keep):
    """Keep only the channels of backbone tapped-layers in `keep` (subset of
    INPUT_LEVEL_LAYERS). The input grid is concat([L4|L6|L8]) on channels in that
    order, equal split. Lets us ablate single levels without rebuilding the cache."""
    if not keep:
        return cache
    C = cache["input_base"].shape[1]
    n = len(INPUT_LEVEL_LAYERS)
    assert C % n == 0, f"input channels {C} not divisible by {n} layers"
    per = C // n
    idxs = []
    for L in keep:
        p = INPUT_LEVEL_LAYERS.index(L)
        idxs.extend(range(p * per, (p + 1) * per))
    sel = torch.tensor(idxs, device=cache["input_base"].device)
    for k in ("input_base", "input_super"):
        cache[k] = cache[k].index_select(1, sel)
    return cache


def load_cache(path, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    # Feature tensors are kept on CPU to avoid GPU OOM for large grids (e.g. 8x8).
    # loss tensors are small and go to device; features are moved per-batch in gather().
    out = {
        "input_base": c["input_base"],   # CPU, keep original dtype (fp16 or fp32)
        "input_super": c["input_super"],
        "loss_base": c["loss_base"].to(device),
        "loss_super": c["loss_super"].to(device),
        "_device": device,
    }
    if "pred_base" in c:
        out["pred_base"] = c["pred_base"]   # CPU
        out["pred_super"] = c["pred_super"]
    return out


def make_loader(cache, batch, shuffle):
    idx = torch.arange(cache["loss_base"].shape[0])
    return DataLoader(TensorDataset(idx), batch_size=batch, shuffle=shuffle)


def gather(cache, idx, prev_action):
    """Select per-sample feature grids by prev_action path (0=base,1=super).
    Grids are 4D [B,C,G,G], so broadcast the mask over the trailing dims.
    Feature tensors live on CPU; move the selected batch to device here."""
    device = cache["_device"]
    idx_cpu = idx.cpu()
    inp_b = cache["input_base"][idx_cpu].to(device, dtype=torch.float32)
    inp_s = cache["input_super"][idx_cpu].to(device, dtype=torch.float32)
    m = prev_action.bool().view(-1, *([1] * (inp_b.dim() - 1)))
    inp = torch.where(m, inp_s, inp_b)
    prd = None
    if "pred_base" in cache:
        prd_b = cache["pred_base"][idx_cpu].to(device, dtype=torch.float32)
        prd_s = cache["pred_super"][idx_cpu].to(device, dtype=torch.float32)
        prd = torch.where(m, prd_s, prd_b)
    return inp, prd


@torch.no_grad()
def val_corr(net, cache, batch_size=512):
    """Pearson corr between predicted A-hat and true advantage A on the val set,
    using prev_action=BASE (path_id=0). Runs in batches to avoid GPU OOM for
    large-grid caches where feature tensors live on CPU."""
    net.eval()
    device = cache["_device"]
    A = (cache["loss_base"].view(-1) - cache["loss_super"].view(-1))
    N = A.shape[0]
    ahs = []
    for start in range(0, N, batch_size):
        idx = slice(start, start + batch_size)
        iv = cache["input_base"][idx].to(device, dtype=torch.float32)
        pv = cache["pred_base"][idx].to(device, dtype=torch.float32) if "pred_base" in cache else None
        pid = torch.zeros(iv.shape[0], dtype=torch.long, device=device)
        ahs.append(net.logit(iv, pv, pid).view(-1))
    ah = torch.cat(ahs)
    a = ah - ah.mean(); b = A - A.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def run_epoch(net, cache, loader, lossfn, opt=None, prev_p=0.5):
    train = opt is not None
    net.train(train)
    agg = {}
    n = 0
    for (idx,) in loader:
        idx = idx.to(cache["loss_base"].device)
        B = idx.shape[0]
        # prev_action ~ Bernoulli(prev_p): fraction of samples whose context comes
        # from the SUPER path (path_id=1). prev_p=0.5 -> 1:1 (default).
        prev = (torch.rand(B, device=idx.device) < prev_p).long()
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
    ap.add_argument("--arch", default="tinyconv", choices=["tinyconv", "gapmpl"],
                    help="router architecture: tinyconv (spatial conv) or gapmpl (GAP then MLP)")
    ap.add_argument("--norm", default="batch", choices=["batch", "layer", "none"],
                    help="GAP normalisation before projection (batch=BatchNorm1d, layer=LayerNorm, none=Identity)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout in the head (regularisation)")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay (L2 regularisation)")
    ap.add_argument("--select", default="val_mse", choices=["val_mse", "val_corr", "last"],
                    help="checkpoint selection: val_mse=min val MSE(A-hat,A); "
                         "val_corr=max val corr(A-hat,A) (ranking, matches eval thresholding); "
                         "last=final epoch (no early stop)")
    ap.add_argument("--mode", default="advantage", choices=["advantage", "bce", "regress"])
    ap.add_argument("--regress_loss", default="mse", choices=["mse", "mae", "huber", "corr"],
                    help="regression loss form (regress mode): mse/mae/huber (magnitude) or corr (ranking)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="bce mode: exclude |A|<margin images from L_acc")
    ap.add_argument("--prev_p", type=float, default=0.5,
                    help="Bernoulli prob that prev_action=SUPER during training (0.5 = 1:1 default)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (weight init + prev_action sampling)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--logdir", default=None, help="directory for per-run training logs")
    ap.add_argument("--tag", default="", help="optional run name suffix for the log file")
    ap.add_argument("--keep_layers", default="", help="comma-sep subset of backbone tapped "
                    "layers to keep (e.g. '4' / '6' / '8'); empty = all. Channel-slices the "
                    "input grid for single-level ablation (no cache rebuild). feat=input only.")
    args = ap.parse_args()
    base = OUT / args.dataset
    if args.cache is None:     args.cache = str(base / "cache_train.pt")
    if args.val_cache is None: args.val_cache = str(base / "cache_val.pt")
    if args.flops is None:     args.flops = str(base / "flops_table.json")
    if args.out is None:       args.out = str(base / "policy.pt")
    # per-run .log/.json are opt-in: only written when --logdir is given (avoids
    # littering outputs/ with regenerable training-history files by default).

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    train_cache = load_cache(args.cache, device)
    val_cache = load_cache(args.val_cache, device) if Path(args.val_cache).exists() else None

    keep = [int(x) for x in args.keep_layers.split(",")] if args.keep_layers else []
    if keep:
        train_cache = slice_levels(train_cache, keep)
        if val_cache is not None:
            val_cache = slice_levels(val_cache, keep)
        print(f"[*] keep_layers={keep} -> input channels {train_cache['input_base'].shape[1]}")

    if args.feat in ("pred", "both") and "pred_base" not in train_cache:
        raise ValueError(f"--feat {args.feat} needs pred (neck) features, but the cache is "
                         f"backbone-only (built with build_cache --feat input). "
                         f"Use --feat input or rebuild the cache with --feat both.")

    # GAP-MLP: pool spatial dims to 1D vectors before training
    if args.arch == "gapmpl":
        for k in list(train_cache.keys()):
            if isinstance(train_cache[k], torch.Tensor) and train_cache[k].dim() == 4:
                train_cache[k] = train_cache[k].mean(dim=(2, 3))
        if val_cache is not None:
            for k in list(val_cache.keys()):
                if isinstance(val_cache[k], torch.Tensor) and val_cache[k].dim() == 4:
                    val_cache[k] = val_cache[k].mean(dim=(2, 3))
        net = GapMlpNet(feat=args.feat).to(device)
    else:
        net = PolicyNetwork(group_dim=args.group_dim, path_dim=args.path_dim, hidden_dim=args.hidden,
                            feat=args.feat, norm=args.norm, dropout=args.dropout).to(device)

    # materialise LazyLinear shapes with a dummy forward (pred optional for
    # backbone-only caches; net ignores it when feat=='input')
    dummy_pred = train_cache["pred_base"][:2].to(device, dtype=torch.float32) if "pred_base" in train_cache else None
    with torch.no_grad():
        net(train_cache["input_base"][:2].to(device, dtype=torch.float32), dummy_pred,
            torch.zeros(2, dtype=torch.long, device=device))

    lossfn = PolicyLoss(args.flops, lambda_flops=args.lambda_flops, lambda_uni=args.lambda_uni,
                        mode=args.mode, margin=args.margin, regress_loss=args.regress_loss)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(train_cache, args.batch, shuffle=True)
    val_loader = make_loader(val_cache, args.batch, shuffle=False) if val_cache else None

    # per-run log file: opt-in via --logdir (None -> stdout only, no files written)
    log_on = bool(args.logdir)
    logpath = None
    if log_on:
        logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_lf{args.lambda_flops}_lu{args.lambda_uni}" + (f"_{args.tag}" if args.tag else "")
        logpath = logdir / f"{name}.log"
        with open(logpath, "w") as f:
            f.write(f"# {json.dumps(vars(args))}\n")
            f.write("epoch\ttotal\tl_acc\tl_flops\tl_uni\tp_super\tval_p_super\tval_l_acc\n")
    history = []

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
        tr = run_epoch(net, train_cache, train_loader, lossfn, opt, prev_p=args.prev_p)
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
        if log_on:
            with open(logpath, "a") as f:
                f.write(f"{ep}\t{tr['total']:.5f}\t{tr['l_acc']:.5f}\t{tr['l_flops']:.5f}\t"
                        f"{tr['l_uni']:.5f}\t{tr['p_super_mean']:.4f}\t"
                        f"{va.get('p_super_mean', float('nan')):.4f}\t{va.get('l_acc', float('nan')):.5f}\n")
    if log_on:
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
