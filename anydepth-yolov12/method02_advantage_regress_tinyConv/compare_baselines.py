"""Compare the learned routing policy against simple cheap heuristics, on images.

All methods are scored apples-to-apples at MATCHED FLOPs: for a target super-rate
r, each method routes its top-(r*N) images (by its own "hardness" score) to SUPER
and the rest to BASE. We then read off the realized detection loss and mean
per-image AP from the cached per-path values. Sweeping r traces each method's
loss/AP-vs-FLOPs curve.

Methods
  oracle      : true advantage  A = loss_base - loss_super     (reachable bound)
  random      : random subset at matched rate                  (lower bound)
  confidence  : route LOW base-path confidence to SUPER        (uncertain = hard)
  luminance   : route DARK images to SUPER                      (night = hard)
  edge        : route HIGH edge-density images to SUPER         (clutter = hard)
  policy      : learned A-hat (mean +/- min/max over the seed checkpoints)

Confidence and AP come from analysis/<dataset>-AnyDepth/per_image_loss_pr_conf.csv
(joined to the cache by image stem). Luminance/edge are computed once from the
val images and cached to outputs/<dataset>/img_proxies.pt.

Usage:
    python method02_advantage_regress_tinyConv/compare_baselines.py --dataset bdd100k
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork

OUT = Path(__file__).resolve().parent / "outputs"


def realized_loss(action, lb, ls, gb, gs):
    a = action.float()
    return ((1 - a) * lb + a * ls).mean().item(), ((1 - a) * gb + a * gs).mean().item()


def realized_ap(action, apb, aps):
    a = action.float()
    return torch.nanmean((1 - a) * apb + a * aps).item()


def topk_action(score, r, N):
    """Route the top (r*N) images by `score` (higher = route to SUPER)."""
    k = int(round(r * N))
    act = torch.zeros(N, dtype=torch.bool)
    if k > 0:
        act[torch.argsort(score, descending=True)[:k]] = True
    return act


def image_proxies(stems, im_files, cache_path, small=256):
    """Per-image (luminance, edge-density), cached to disk keyed by stem."""
    import cv2
    import numpy as np
    cp = Path(cache_path)
    cached = torch.load(cp, weights_only=False) if cp.exists() else {}
    lum, edge = {}, {}
    missing = [(s, f) for s, f in zip(stems, im_files) if s not in cached]
    for i, (s, f) in enumerate(missing):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            cached[s] = (float("nan"), float("nan")); continue
        h, w = g.shape
        g = cv2.resize(g, (small, int(small * h / w)))
        e = cv2.Canny(g, 100, 200)
        cached[s] = (float(g.mean()), float((e > 0).mean()))
        if i % 1000 == 0:
            print(f"  proxies {i}/{len(missing)}")
    if missing:
        torch.save(cached, cp); print(f"[*] image proxies -> {cp}")
    lum = torch.tensor([cached[s][0] for s in stems])
    edge = torch.tensor([cached[s][1] for s in stems])
    return lum, edge


@torch.no_grad()
def policy_ahat(ckpt_path, cache, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 64), feat=a.get("feat", "both"),
                        norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(device)
    net.eval()
    inp = cache["input_base"].to(device)
    prd = cache.get("pred_base")
    prd = prd.to(device) if prd is not None else None
    pid = torch.zeros(inp.shape[0], dtype=torch.long, device=device)
    net(inp[:2], None if prd is None else prd[:2], pid[:2])  # materialise lazy layers
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net.logit(inp, prd, pid).view(-1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bdd100k")
    ap.add_argument("--policies", nargs="*", default=None,
                    help="policy ckpts to overlay (default: outputs/<ds>/policy_*.pt)")
    ap.add_argument("--val_cache", default=None)
    ap.add_argument("--flops", default=None)
    ap.add_argument("--ap_csv", default=None)
    ap.add_argument("--conf_col", default="mean_conf_all_base",
                    help="base-path confidence column to route on")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--points", type=int, default=21, help="super-rate sweep resolution")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    base = OUT / args.dataset
    if args.val_cache is None: args.val_cache = str(base / "cache_val.pt")
    if args.flops is None:     args.flops = str(base / "flops_table.json")
    if args.ap_csv is None:    args.ap_csv = f"analysis/{args.dataset}-AnyDepth/per_image_loss_pr_conf.csv"
    if args.policies is None:  args.policies = sorted(str(p) for p in base.glob("policy_*.pt"))
    if args.out is None:       args.out = str(base / "eval" / "baselines.json")
    if args.plot is None:      args.plot = str(base / "eval" / "baselines.png")

    device = args.device if torch.cuda.is_available() else "cpu"
    table = json.loads(Path(args.flops).read_text())
    gb = table["actions"]["0_base"]["gflops"]
    gs = table["actions"]["1_super"]["gflops"]

    c = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    lb, ls = c["loss_base"].view(-1), c["loss_super"].view(-1)
    N = lb.shape[0]
    stems = [Path(p).stem for p in c["im_file"]]

    # confidence + AP from the per-image csv (joined by stem)
    import pandas as pd
    m = pd.read_csv(args.ap_csv).set_index("stem")
    miss = [s for s in stems if s not in m.index]
    if miss:
        raise SystemExit(f"{len(miss)} cache stems absent from {args.ap_csv}")
    conf = torch.tensor(m.loc[stems, args.conf_col].to_numpy(), dtype=torch.float)
    apb = torch.tensor(m.loc[stems, "ap5095_base"].to_numpy(), dtype=torch.float)
    aps = torch.tensor(m.loc[stems, "ap5095_super"].to_numpy(), dtype=torch.float)

    lum, edge = image_proxies(stems, c["im_file"], base / "img_proxies.pt")

    # scores: higher = route to SUPER ("harder")
    scores = {
        "oracle":     (lb - ls),
        "confidence": -conf,        # low base confidence -> super
        "luminance":  -lum,         # dark -> super
        "edge":       edge,         # high edge density -> super
    }
    policy_scores = [policy_ahat(p, c, device) for p in args.policies]

    rates = torch.linspace(0, 1, args.points).tolist()
    torch.manual_seed(0)
    rand_score = torch.rand(N)

    def curve(score):
        rows = []
        for r in rates:
            act = topk_action(score, r, N)
            loss, fl = realized_loss(act, lb, ls, gb, gs)
            rows.append({"rate": r, "gflops": fl, "loss": loss,
                         "ap": realized_ap(act, apb, aps)})
        return rows

    out = {"refs": {
        "base":  {"gflops": gb, "loss": lb.mean().item(), "ap": apb.nanmean().item()},
        "super": {"gflops": gs, "loss": ls.mean().item(), "ap": aps.nanmean().item()}},
        "methods": {}}
    out["methods"]["random"] = curve(rand_score)
    for name, sc in scores.items():
        out["methods"][name] = curve(sc)
    out["methods"]["policy_seeds"] = [curve(sc) for sc in policy_scores]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[*] saved -> {args.out}")

    # console summary at ~50% super-rate (the middle sweep point)
    mid = args.points // 2
    print(f"\n{'method':<14}{'GFLOPs':>9}{'loss':>9}{'AP':>9}  (@ super-rate {rates[mid]:.2f})")
    for name in ["oracle", "policy", "confidence", "edge", "luminance", "random"]:
        if name == "policy":
            aps_mid = [curve(sc)[mid] for sc in policy_scores]
            apv = sum(p["ap"] for p in aps_mid) / len(aps_mid)
            lo = sum(p["loss"] for p in aps_mid) / len(aps_mid)
            fl = aps_mid[0]["gflops"]
        else:
            row = out["methods"][name][mid]
            apv, lo, fl = row["ap"], row["loss"], row["gflops"]
        print(f"{name:<14}{fl:>9.2f}{lo:>9.4f}{apv:>9.4f}")

    if args.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        colors = {"oracle": "tab:green", "random": "tab:gray", "confidence": "tab:blue",
                  "luminance": "tab:orange", "edge": "tab:purple"}
        for metric, ylab, better in [("ap", "mean per-image AP@[.5:.95]", "higher"),
                                     ("loss", "detection loss", "lower")]:
            plt.figure(figsize=(8, 5.5))
            for name in ["oracle", "confidence", "edge", "luminance", "random"]:
                rows = sorted(out["methods"][name], key=lambda r: r["gflops"])
                style = "s--" if name == "oracle" else ("^:" if name == "random" else "o-")
                plt.plot([r["gflops"] for r in rows], [r[metric] for r in rows],
                         style, color=colors[name], label=name, alpha=0.9, ms=4)
            # policy: mean +/- min/max band over seeds
            gf = [r["gflops"] for r in out["methods"]["policy_seeds"][0]]
            ys = np.array([[r[metric] for r in s] for s in out["methods"]["policy_seeds"]])
            plt.plot(gf, ys.mean(0), "o-", color="tab:red", lw=2,
                     label=f"policy (mean of {ys.shape[0]})", ms=4)
            plt.fill_between(gf, ys.min(0), ys.max(0), color="tab:red", alpha=0.2)
            for y, mk, cl in [(out["refs"]["base"][metric], "*", "darkgreen"),
                              (out["refs"]["super"][metric], "*", "black")]:
                x = out["refs"]["base"]["gflops"] if cl == "darkgreen" else out["refs"]["super"]["gflops"]
                plt.scatter([x], [y], marker=mk, s=200, color=cl, zorder=5)
            if better == "lower":
                plt.gca().invert_yaxis()
            plt.xlabel("GFLOPs (per frame)")
            plt.ylabel(f"{ylab} ({better}=better)")
            plt.title(f"Static image routing: {metric.upper()} vs compute ({args.dataset})")
            plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
            pp = Path(args.plot); pp.parent.mkdir(parents=True, exist_ok=True)
            fp = pp.with_name(f"{pp.stem}_{metric}.png")
            plt.savefig(fp, dpi=150); print(f"[*] plot -> {fp}")


if __name__ == "__main__":
    main()
