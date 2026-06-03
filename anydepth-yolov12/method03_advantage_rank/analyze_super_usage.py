"""Analysis #4: what does the policy route to SUPER, and how close is it to oracle?

Uses the offline val cache (no detector forward) + a trained backbone policy:
  - A-hat per image (mean over the 5 backbone seeds, path_id=0)
  - true advantage A = loss_base - loss_super
  - scene attributes from KITTI YOLO GT labels: object count, mean box area
Produces:
  - correlation table (A-hat vs A, vs n_obj, vs mean_area; and A vs attrs)
  - scatter plots (A-hat vs n_obj, A-hat vs mean_area)
  - top-k / bottom-k A-hat image stems (qualitative: which frames want SUPER)
  - oracle gap: policy loss-vs-FLOPs curve vs per-image oracle front

Usage:
    python method03_advantage_rank/analyze_super_usage.py --dataset kitti
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from method03_advantage_rank.policy_net import PolicyNetwork

OUT = Path(__file__).resolve().parent / "outputs"


def load_policy(path, c, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get("args", {})
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 128), feat=a.get("feat", "input"),
                        norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(device)
    net.eval()
    with torch.no_grad():
        net(c["input_base"][:2].to(device), c["pred_base"][:2].to(device),
            torch.zeros(2, dtype=torch.long, device=device))
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net


@torch.no_grad()
def ahat(net, c, device):
    pid = torch.zeros(c["input_base"].shape[0], dtype=torch.long, device=device)
    return net.logit(c["input_base"].to(device), c["pred_base"].to(device), pid).view(-1).cpu()


def scene_attrs(im_files, label_dir):
    """per image: (n_objects, mean_box_area) from YOLO labels (area in [0,1])."""
    n_obj, mean_area = [], []
    for p in im_files:
        lf = label_dir / (Path(p).stem + ".txt")
        if not lf.exists():
            n_obj.append(0); mean_area.append(0.0); continue
        areas = []
        for ln in lf.read_text().splitlines():
            t = ln.split()
            if len(t) >= 5:
                areas.append(float(t[3]) * float(t[4]))  # w*h (normalised)
        n_obj.append(len(areas))
        mean_area.append(float(np.mean(areas)) if areas else 0.0)
    return np.array(n_obj, float), np.array(mean_area, float)


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--policy_glob", default="ablation/policy_input_s*.pt",
                    help="backbone policies to average A-hat over (relative to outputs/<dataset>/)")
    ap.add_argument("--label_dir", default="/media/data/kitti_yolo/labels/val")
    ap.add_argument("--flops", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    base = OUT / args.dataset
    if args.flops is None:
        args.flops = str(base / "flops_table.json")
    device = args.device if torch.cuda.is_available() else "cpu"
    outdir = base / "eval" / "analysis"; outdir.mkdir(parents=True, exist_ok=True)

    c = torch.load(base / "cache_val.pt", map_location="cpu", weights_only=False)
    lb, ls = c["loss_base"].view(-1), c["loss_super"].view(-1)
    A = (lb - ls).numpy()
    im = c["im_file"]; N = len(im)

    # mean A-hat over backbone seeds
    pols = sorted(base.glob(args.policy_glob))
    assert pols, f"no policies match {args.policy_glob}"
    ah = torch.stack([ahat(load_policy(p, c, device), c, device) for p in pols]).mean(0).numpy()
    print(f"[*] averaged A-hat over {len(pols)} policies")

    n_obj, mean_area = scene_attrs(im, Path(args.label_dir))

    # ---- correlations ----
    table = {
        "corr(Ahat, A)": corr(ah, A),
        "corr(Ahat, n_obj)": corr(ah, n_obj),
        "corr(Ahat, mean_area)": corr(ah, mean_area),
        "corr(A, n_obj)": corr(A, n_obj),
        "corr(A, mean_area)": corr(A, mean_area),
    }
    print("\n=== correlations ===")
    for k, v in table.items():
        print(f"  {k:<24} {v:+.3f}")

    # ---- oracle gap (static cache) ----
    t = json.loads(Path(args.flops).read_text())
    gb = t["actions"]["0_base"]["gflops"]; gs = t["actions"]["1_super"]["gflops"]
    order = np.argsort(-A)            # images that benefit most from SUPER first
    rates = np.linspace(0, 1, 11)
    oracle_loss, policy_loss = [], []
    qs = np.quantile(ah, np.linspace(0, 1, 11))
    for r, q in zip(rates, qs):
        k = int(round(r * N))
        oa = np.zeros(N, bool); oa[order[:k]] = True
        oracle_loss.append(float((np.where(oa, ls.numpy(), lb.numpy())).mean()))
        pa = ah >= q
        policy_loss.append(float((np.where(pa, ls.numpy(), lb.numpy())).mean()))

    # ---- top/bottom A-hat stems ----
    idx = np.argsort(-ah)
    top = [(Path(im[i]).stem, float(ah[i]), int(n_obj[i]), float(mean_area[i])) for i in idx[:args.topk]]
    bot = [(Path(im[i]).stem, float(ah[i]), int(n_obj[i]), float(mean_area[i])) for i in idx[-args.topk:]]

    # ---- save json + md ----
    res = {"n_images": N, "n_policies": len(pols), "correlations": table,
           "oracle_gap": {"rate": rates.tolist(), "oracle_loss": oracle_loss,
                          "policy_loss": policy_loss},
           "top_ahat": top, "bottom_ahat": bot,
           "means": {"top_n_obj": float(np.mean([t[2] for t in top])),
                     "bot_n_obj": float(np.mean([t[2] for t in bot])),
                     "top_mean_area": float(np.mean([t[3] for t in top])),
                     "bot_mean_area": float(np.mean([t[3] for t in bot]))}}
    (outdir / "super_usage.json").write_text(json.dumps(res, indent=2))
    print(f"\n[*] top-{args.topk} A-hat (wants SUPER): mean n_obj={res['means']['top_n_obj']:.1f} "
          f"mean_area={res['means']['top_mean_area']:.4f}")
    print(f"[*] bot-{args.topk} A-hat (wants BASE) : mean n_obj={res['means']['bot_n_obj']:.1f} "
          f"mean_area={res['means']['bot_mean_area']:.4f}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].scatter(n_obj, ah, s=8, alpha=0.4, color="tab:red")
    ax[0].set_xlabel("# GT objects"); ax[0].set_ylabel("A-hat (predicted advantage)")
    ax[0].set_title(f"A-hat vs object count (r={table['corr(Ahat, n_obj)']:+.2f})")
    ax[1].scatter(mean_area, ah, s=8, alpha=0.4, color="tab:blue")
    ax[1].set_xlabel("mean GT box area (norm)"); ax[1].set_ylabel("A-hat")
    ax[1].set_xscale("log")
    ax[1].set_title(f"A-hat vs mean box area (r={table['corr(Ahat, mean_area)']:+.2f})")
    ax[2].plot([(r) * 100 for r in rates], policy_loss, "o-", color="tab:red", label="policy")
    ax[2].plot([(r) * 100 for r in rates], oracle_loss, "s--", color="tab:green", label="oracle")
    ax[2].set_xlabel("SUPER usage (%)"); ax[2].set_ylabel("mean detection loss")
    ax[2].invert_yaxis(); ax[2].legend(); ax[2].set_title("oracle gap (static cache)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fp = outdir / "super_usage.png"
    fig.savefig(fp, dpi=150)
    print(f"[*] saved -> {fp}\n[*] json   -> {outdir/'super_usage.json'}")


if __name__ == "__main__":
    main()
