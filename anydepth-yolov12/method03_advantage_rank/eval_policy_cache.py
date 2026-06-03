"""Cache-based evaluation of a trained policy (method01, path B).

Without any detector forward or video, this gives the static-image analog of
the AP-vs-FLOPs curve directly from the per-image cached losses:

  for a threshold t: action_i = SUPER if p_super_i >= t else BASE
    realized loss  = mean_i  ( loss_base_i if BASE else loss_super_i )
    realized GFLOPs= mean_i  ( g_base      if BASE else g_super      )

Sweeping t traces the policy's loss-vs-FLOPs curve. Reference points:
  - always-BASE / always-SUPER : the two endpoints
  - ORACLE  : per-image argmin(loss_base, loss_super) -> reachable lower bound
  - RANDOM  : random path per image at a matched super-rate -> upper bound

A good policy curve hugs the ORACLE front and beats RANDOM at equal FLOPs.

Usage:
    # kitti
    python method03_advantage_rank/eval_policy_cache.py --dataset kitti

    # bdd100k  (uses outputs/bdd100k/{policy,cache_val,flops_table}; static-image
    # AP-vs-FLOPs only -- there is no tracking video, so eval_video.py does not
    # apply to bdd100k)
    python method03_advantage_rank/eval_policy_cache.py --dataset bdd100k
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from method03_advantage_rank.policy_net import PolicyNetwork


def realized(action, lb, ls, gb, gs):
    """action: bool tensor (True=SUPER). Returns (mean_loss, mean_gflops)."""
    a = action.float()
    loss = ((1 - a) * lb + a * ls).mean().item()
    flops = ((1 - a) * gb + a * gs).mean().item()
    return loss, flops


def realized_ap(action, apb, aps):
    """Mean per-image AP under a routing decision (NaN/no-GT images ignored)."""
    a = action.float()
    return torch.nanmean((1 - a) * apb + a * aps).item()


@torch.no_grad()
def policy_scores(net, cache, device):
    """Routing score per image, using prev_action=BASE features (path_id=0).
    Uses the raw logit (= predicted advantage A-hat for regress models) so the
    threshold sweep has full resolution; sigmoid would compress it near 0.5."""
    inp = cache["input_base"].to(device)
    prd = cache["pred_base"].to(device)
    pid = torch.zeros(inp.shape[0], dtype=torch.long, device=device)
    return net.logit(inp, prd, pid).view(-1).cpu()


OUT = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti", help="output scope: outputs/<dataset>/")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--val_cache", default=None)
    ap.add_argument("--flops", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default=None, help="path for loss-vs-FLOPs figure ('' to disable)")
    ap.add_argument("--ap_csv", default=None,
                    help="per_image_loss_pr_conf.csv with ap5095_{super,base} per stem; "
                         "when given, also plots mean per-image AP vs FLOPs "
                         "(default: analysis/<dataset>-AnyDepth/per_image_loss_pr_conf.csv if present)")
    args = ap.parse_args()
    base = OUT / args.dataset
    if args.policy is None:    args.policy = str(base / "policy.pt")
    if args.val_cache is None: args.val_cache = str(base / "cache_val.pt")
    if args.flops is None:     args.flops = str(base / "flops_table.json")
    if args.plot is None:      args.plot = str(base / "eval" / "cache_curve.png")
    if args.ap_csv is None:
        cand = Path(f"analysis/{args.dataset}-AnyDepth/per_image_loss_pr_conf.csv")
        args.ap_csv = str(cand) if cand.exists() else ""

    device = args.device if torch.cuda.is_available() else "cpu"
    table = json.loads(Path(args.flops).read_text())
    gb = table["actions"]["0_base"]["gflops"]
    gs = table["actions"]["1_super"]["gflops"]

    c = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    lb, ls = c["loss_base"].view(-1), c["loss_super"].view(-1)
    N = lb.shape[0]

    # optional per-image AP, aligned to cache order via stem join
    apb = aps = None
    if args.ap_csv and Path(args.ap_csv).exists():
        import pandas as pd
        m = pd.read_csv(args.ap_csv).set_index("stem")
        stems = [Path(p).stem for p in c["im_file"]]
        missing = [s for s in stems if s not in m.index]
        if missing:
            print(f"[!] {len(missing)} cache stems not in ap_csv; skipping AP plot")
        else:
            aps = torch.tensor(m.loc[stems, "ap5095_super"].to_numpy(), dtype=torch.float)
            apb = torch.tensor(m.loc[stems, "ap5095_base"].to_numpy(), dtype=torch.float)
            # NaN (images with no GT) -> drop from AP means by treating as not-counted
            print(f"[*] AP join ok: mean AP base={apb.nanmean():.4f} super={aps.nanmean():.4f}")

    ckpt = torch.load(args.policy, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 64), feat=a.get("feat", "both"),
                        norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(device)
    net.eval()
    with torch.no_grad():  # materialise LazyLinear (eval mode for BN with batch 2 ok)
        net(c["input_base"][:2].to(device), c["pred_base"][:2].to(device),
            torch.zeros(2, dtype=torch.long, device=device))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    scores = policy_scores(net, {"input_base": c["input_base"], "pred_base": c["pred_base"]}, device)

    # references
    base_loss, base_flops = realized(torch.zeros(N, dtype=torch.bool), lb, ls, gb, gs)
    super_loss, super_flops = realized(torch.ones(N, dtype=torch.bool), lb, ls, gb, gs)
    oracle_action = ls < lb  # SUPER only where it helps
    oracle_loss, oracle_flops = realized(oracle_action, lb, ls, gb, gs)

    adv = (lb - ls)
    order = torch.argsort(adv, descending=True)  # images that benefit most from SUPER first
    # AP-gain ordering (independent oracle front for the AP metric)
    if apb is not None:
        ap_gain = torch.nan_to_num(aps - apb, nan=0.0)
        ap_order = torch.argsort(ap_gain, descending=True)
        base_ap = realized_ap(torch.zeros(N, dtype=torch.bool), apb, aps)
        super_ap = realized_ap(torch.ones(N, dtype=torch.bool), apb, aps)
        oracle_ap = realized_ap(ap_gain > 0, apb, aps)

    print(f"{'point':<14}{'super_rate':>11}{'GFLOPs':>9}{'loss':>9}")
    print(f"{'always-BASE':<14}{0.0:>11.3f}{base_flops:>9.2f}{base_loss:>9.4f}")
    print(f"{'always-SUPER':<14}{1.0:>11.3f}{super_flops:>9.2f}{super_loss:>9.4f}")
    print(f"{'ORACLE':<14}{oracle_action.float().mean().item():>11.3f}{oracle_flops:>9.2f}{oracle_loss:>9.4f}")
    print("-" * 43)
    if apb is not None:
        print(f"[AP refs] base={base_ap:.4f}  super={super_ap:.4f}  oracle(AP-gain)={oracle_ap:.4f}")

    def oracle_front(super_rate):
        """Best achievable loss at a given super-rate: flip the top-A images to SUPER."""
        k = int(round(super_rate * N))
        act = torch.zeros(N, dtype=torch.bool)
        act[order[:k]] = True
        return realized(act, lb, ls, gb, gs)

    def oracle_ap_front(super_rate):
        k = int(round(super_rate * N))
        act = torch.zeros(N, dtype=torch.bool)
        act[ap_order[:k]] = True
        return realized_ap(act, apb, aps)

    # sweep thresholds over the score distribution quantiles (full resolution)
    qs = torch.quantile(scores, torch.linspace(0, 1, 21)).tolist()
    print(f"{'t':<8}{'super_rate':>11}{'GFLOPs':>9}{'loss':>9}{'oracle@FLOPs':>14}{'random':>9}")
    curve = []
    for t in qs:
        action = scores >= t
        sr = action.float().mean().item()
        loss, flops = realized(action, lb, ls, gb, gs)
        o_loss, _ = oracle_front(sr)
        torch.manual_seed(0)
        rnd = torch.rand(N) < sr
        r_loss, _ = realized(rnd, lb, ls, gb, gs)
        pt = {"thres": t, "super_rate": sr, "gflops": flops, "loss": loss,
              "oracle_loss": o_loss, "rand_loss": r_loss}
        if apb is not None:
            pt["ap"] = realized_ap(action, apb, aps)
            pt["oracle_ap"] = oracle_ap_front(sr)
            pt["rand_ap"] = realized_ap(rnd, apb, aps)
        curve.append(pt)
        print(f"t={t:<6.2f}{sr:>11.3f}{flops:>9.2f}{loss:>9.4f}{o_loss:>14.4f}{r_loss:>9.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"refs": {"base": [base_flops, base_loss], "super": [super_flops, super_loss],
                      "oracle": [oracle_flops, oracle_loss]}, "curve": curve}, indent=2))
        print(f"[*] saved -> {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cs = sorted(curve, key=lambda r: r["gflops"])
        gf = [r["gflops"] for r in cs]
        plt.figure(figsize=(8, 5.5))
        plt.plot(gf, [r["loss"] for r in cs], "o-", color="tab:red", label="policy (A-hat)")
        plt.plot(gf, [r["oracle_loss"] for r in cs], "s--", color="tab:green",
                 label="oracle @ matched FLOPs")
        plt.plot(gf, [r["rand_loss"] for r in cs], "^:", color="tab:gray", label="random")
        for x, y, t in [(r["gflops"], r["loss"], r["thres"]) for r in cs]:
            plt.annotate(f"{t:g}", (x, y), textcoords="offset points", xytext=(0, 4),
                         fontsize=6, color="tab:red", ha="center")
        plt.scatter([base_flops], [base_loss], marker="*", s=200, color="darkgreen", zorder=5)
        plt.annotate("base", (base_flops, base_loss), textcoords="offset points", xytext=(6, 6))
        plt.scatter([super_flops], [super_loss], marker="*", s=200, color="black", zorder=5)
        plt.annotate("super", (super_flops, super_loss), textcoords="offset points", xytext=(6, 6))
        plt.axhline(super_loss, color="black", ls=":", alpha=0.4)
        plt.gca().invert_yaxis()  # lower detection loss is better -> up
        plt.xlabel("GFLOPs (per frame)"); plt.ylabel("detection loss (lower=better)")
        plt.title("Cache (static) eval: detection loss vs compute")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        pp = Path(args.plot); pp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pp, dpi=150); print(f"[*] plot -> {pp}")

        # second figure: mean per-image AP vs FLOPs (higher = better)
        if apb is not None:
            plt.figure(figsize=(8, 5.5))
            plt.plot(gf, [r["ap"] for r in cs], "o-", color="tab:red", label="policy (A-hat)")
            plt.plot(gf, [r["oracle_ap"] for r in cs], "s--", color="tab:green",
                     label="oracle @ matched FLOPs (AP-gain)")
            plt.plot(gf, [r["rand_ap"] for r in cs], "^:", color="tab:gray", label="random")
            plt.scatter([base_flops], [base_ap], marker="*", s=200, color="darkgreen", zorder=5)
            plt.annotate("base", (base_flops, base_ap), textcoords="offset points", xytext=(6, -10))
            plt.scatter([super_flops], [super_ap], marker="*", s=200, color="black", zorder=5)
            plt.annotate("super", (super_flops, super_ap), textcoords="offset points", xytext=(6, 6))
            plt.xlabel("GFLOPs (per frame)")
            plt.ylabel("mean per-image AP@[.5:.95] (higher=better)")
            plt.title("Cache (static) eval: mean per-image AP vs compute")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            app = pp.with_name(pp.stem + "_ap.png")
            plt.savefig(app, dpi=150); print(f"[*] AP plot -> {app}")


if __name__ == "__main__":
    main()
