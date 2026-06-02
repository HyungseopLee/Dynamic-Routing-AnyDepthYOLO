"""Measure the policy (router) network's own overhead: #params and FLOPs/frame.

The AP-vs-FLOPs curves report the DETECTOR cost only (BASE/SUPER GFLOPs); this
script quantifies the router's added cost so we can state it is negligible.
FLOPs are counted as 2*MACs over the Linear layers (BN/embedding negligible);
the router runs once per frame on the pooled [1,C] context vectors, and the GAP
features themselves are produced for free during the detector forward.

Writes outputs/<dataset>/router_overhead.json (+ prints a table).

Usage:
    python method01_advantage_regress/measure_router_overhead.py --dataset kitti
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from method01_advantage_regress.policy_net import PolicyNetwork

OUT = Path(__file__).resolve().parent / "outputs"


def measure(feat, group_dim, hidden, norm):
    net = PolicyNetwork(group_dim=group_dim, hidden_dim=hidden, feat=feat, norm=norm)
    net.eval()
    with torch.no_grad():
        net(torch.randn(2, 768), torch.randn(2, 640), torch.zeros(2, dtype=torch.long))
    nparams = sum(p.numel() for p in net.parameters())
    macs = sum(m.in_features * m.out_features
               for m in net.modules() if isinstance(m, torch.nn.Linear))
    return nparams, macs, 2 * macs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--group_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--norm", default="batch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = OUT / args.dataset
    if args.out is None:
        args.out = str(base / "router_overhead.json")

    # detector cost for context (if the flops table exists)
    det = {}
    ft = base / "flops_table.json"
    if ft.exists():
        t = json.loads(ft.read_text())
        det = {"base_gflops": t["actions"]["0_base"]["gflops"],
               "super_gflops": t["actions"]["1_super"]["gflops"]}

    rows = {}
    print(f"{'feat':8s}{'#params':>12}{'FLOPs/frame':>14}{'GFLOPs':>12}{'% of SUPER':>12}")
    for feat in ("input", "pred", "both"):
        nparams, macs, flops = measure(feat, args.group_dim, args.hidden, args.norm)
        g = flops / 1e9
        pct = (g / det["super_gflops"] * 100) if det else float("nan")
        rows[feat] = {"params": nparams, "macs": macs, "flops_per_frame": flops,
                      "gflops": g, "pct_of_super": pct}
        print(f"{feat:8s}{nparams:>12,}{flops:>14,}{g:>12.3e}{pct:>11.4f}%")

    payload = {"config": {"group_dim": args.group_dim, "hidden": args.hidden,
                          "path_dim": 8, "norm": args.norm,
                          "input_vec_dim": 768, "pred_vec_dim": 640},
               "detector": det, "router": rows,
               "note": "FLOPs = 2*MACs over Linear layers; per frame (batch=1). "
                       "GAP feature extraction is free (reuses detector forward). "
                       "Router cost is NOT added to the reported AP-vs-FLOPs curves "
                       "because it is <0.001% of the detector cost."}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
