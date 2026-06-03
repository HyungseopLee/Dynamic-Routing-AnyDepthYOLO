"""Derive per-policy decision thresholds on the VALIDATION cache (offline).

Honest deployment pipeline: the routing threshold tau must be chosen WITHOUT
looking at the test (video) set. This script loads trained policies + the val
cache, computes A-hat per val frame (path_id=0), and for each target SUPER
budget r picks tau_r = quantile(A-hat, 1-r) so that a fraction r of val frames
would route to SUPER. eval_video then APPLIES these fixed thresholds on video
(via --val_taus_json); it never tunes tau on the video set.

Writes (evidence):
  outputs/<dataset>/eval/<name>.json  {meta, thresholds[ptag][budget]=tau, ahat_stats}
  outputs/<dataset>/eval/<name>.log   datetime / args / per-policy A-hat stats / tau table

Usage:
    python method01_advantage_regress/get_thresholds.py --dataset kitti \
        --policy_glob 'ablation/policy_input_s*.pt' --name val_thresholds_tinyconv_backbone
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from method01_advantage_regress.policy_net import PolicyNetwork

OUT = Path(__file__).resolve().parent / "outputs"


def load_policy(path, c, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get("args", {})
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 128), feat=a.get("feat", "input"),
                        norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(device)
    net.eval()
    with torch.no_grad():  # build Lazy layers
        net(c["input_base"][:2].to(device), c["pred_base"][:2].to(device),
            torch.zeros(2, dtype=torch.long, device=device))
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net


@torch.no_grad()
def ahat(net, c, device):
    pid = torch.zeros(c["input_base"].shape[0], dtype=torch.long, device=device)
    return net.logit(c["input_base"].to(device), c["pred_base"].to(device), pid).view(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--policy_glob", default="ablation/policy_input_s*.pt",
                    help="policies (relative to outputs/<dataset>/); tag = file stem")
    ap.add_argument("--budgets", default="10,20,30,40,50,60,70,80,90",
                    help="comma-sep target SUPER-rate %%")
    ap.add_argument("--name", default="val_thresholds",
                    help="output basename under outputs/<dataset>/eval/")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    base = OUT / args.dataset
    device = args.device if torch.cuda.is_available() else "cpu"
    budgets = [int(b) for b in args.budgets.split(",")]
    c = torch.load(base / "cache_val.pt", map_location="cpu", weights_only=False)
    N = c["input_base"].shape[0]

    pols = sorted(base.glob(args.policy_glob))
    assert pols, f"no policies match {args.policy_glob}"

    thresholds, stats = {}, {}
    lines = [f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] get_thresholds",
             f"dataset={args.dataset}  val_frames={N}  budgets={budgets}",
             f"policy_glob={args.policy_glob}  ({len(pols)} policies)", ""]
    for p in pols:
        ptag = p.stem.replace("policy_", "")
        ah = ahat(load_policy(p, c, device), c, device)
        thresholds[ptag] = {b: float(np.quantile(ah, 1.0 - b / 100.0)) for b in budgets}
        stats[ptag] = {"mean": float(ah.mean()), "std": float(ah.std()),
                       "min": float(ah.min()), "max": float(ah.max())}
        lines.append(f"[{ptag}] A-hat mean={ah.mean():+.4f} std={ah.std():.4f} "
                     f"range=[{ah.min():+.3f},{ah.max():+.3f}]")
        lines.append("  budget(%) -> tau: " +
                     "  ".join(f"{b}:{thresholds[ptag][b]:+.4f}" for b in budgets))

    eval_dir = base / "eval"; eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"dataset": args.dataset, "val_frames": N, "budgets": budgets,
                        "policy_glob": args.policy_glob, "n_policies": len(pols),
                        "datetime": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"},
               "thresholds": thresholds, "ahat_stats": stats}
    jpath = eval_dir / f"{args.name}.json"
    lpath = eval_dir / f"{args.name}.log"
    jpath.write_text(json.dumps(payload, indent=2))
    lpath.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[*] thresholds -> {jpath}\n[*] log        -> {lpath}")


if __name__ == "__main__":
    main()
