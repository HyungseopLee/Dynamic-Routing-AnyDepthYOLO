"""Measure the policy (router) network's own overhead: #params and FLOPs/frame.

method02 router (tiny-conv): the group projection keeps spatial structure, so we
count MACs over BOTH Conv2d and Linear layers via forward hooks (using each
layer's actual output spatial size). The router runs once per frame on the small
GxG feature grids, which are produced for free during the detector forward.

Writes outputs/<dataset>/router_overhead.json (+ prints a table).

Usage:
    python method02_advantage_regress_tinyConv/measure_router_overhead.py --dataset kitti
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork

OUT = Path(__file__).resolve().parent / "outputs"


def measure(feat, group_dim, hidden, norm, grid=4):
    net = PolicyNetwork(group_dim=group_dim, hidden_dim=hidden, feat=feat, norm=norm)
    net.eval()
    G = grid
    inp = torch.randn(1, 768, G, G)
    prd = torch.randn(1, 640, G, G)
    pid = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        net(inp, prd, pid)  # build Lazy layers

    # count MACs via hooks on the actual (batch=1) forward
    macs_total = [0]

    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            out_elems = o.shape[1] * o.shape[2] * o.shape[3]      # Cout*H*W
            macs_total[0] += out_elems * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear):
            macs_total[0] += m.in_features * m.out_features

    handles = [m.register_forward_hook(hook)
               for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        net(inp, prd, pid)
    for h in handles:
        h.remove()

    nparams = sum(p.numel() for p in net.parameters())
    macs = macs_total[0]
    return nparams, macs, 2 * macs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--group_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--grid", type=int, default=4)
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
        nparams, macs, flops = measure(feat, args.group_dim, args.hidden, args.norm, args.grid)
        g = flops / 1e9
        pct = (g / det["super_gflops"] * 100) if det else float("nan")
        rows[feat] = {"params": nparams, "macs": macs, "flops_per_frame": flops,
                      "gflops": g, "pct_of_super": pct}
        print(f"{feat:8s}{nparams:>12,}{flops:>14,}{g:>12.3e}{pct:>11.4f}%")

    payload = {"config": {"group_dim": args.group_dim, "hidden": args.hidden,
                          "grid": args.grid, "path_dim": 8, "norm": args.norm,
                          "input_vec_dim": 768, "pred_vec_dim": 640},
               "detector": det, "router": rows,
               "note": "FLOPs = 2*MACs over Conv2d + Linear layers (tiny-conv router); "
                       "per frame (batch=1) on GxG feature grids. Feature-grid pooling "
                       "is free (reuses detector forward). Router cost is NOT added to "
                       "the reported AP-vs-FLOPs curves because it is <0.01% of detector."}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
