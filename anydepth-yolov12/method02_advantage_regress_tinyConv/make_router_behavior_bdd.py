"""Router behavior analysis on BDD100K val.

Groups validation frames by the router's predicted advantage A-hat into deciles and
characterizes each decile with INTERPRETABLE scene attributes (movable-object count,
small-object fraction, scene luminance, edge density, night fraction). Also reports
how strongly A-hat correlates with each interpretable scalar -- and contrasts that
with the true advantage A -- to show the router keys on fine image-level factors
rather than any single semantic/appearance heuristic.

    conda run -n yolov12 python -m method02_advantage_regress_tinyConv.make_router_behavior_bdd \
        --out_dir ../paper/_extracted
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork  # noqa

EXCLUDE = {"traffic sign", "traffic light"}   # count movable objects only
SMALL_AREA = 32 * 32                           # COCO "small" threshold (px^2)


def load_router(path, cache, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 64), feat=a.get("feat", "input"),
                        norm=a.get("norm", "batch")).to(device)
    net.eval()
    ci = cache["input_base"].shape[1]; g = cache["input_base"].shape[-1]
    with torch.no_grad():
        net(torch.zeros(2, ci, g, g, device=device), None,
            torch.zeros(2, dtype=torch.long, device=device))
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, a.get("feat", "input")


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-8))


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="method02_advantage_regress_tinyConv/outputs/bdd100k/cache_val_g2.pt")
    ap.add_argument("--policy", default="method02_advantage_regress_tinyConv/outputs/bdd100k/policy_scenario_s0.pt")
    ap.add_argument("--labels_json", default="/media/data/bdd100k_yolo/bdd100k_labels_images_val.json")
    ap.add_argument("--out_dir", default=str(ROOT.parent / "paper/_extracted"))
    ap.add_argument("--nbins", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    cache = torch.load(args.cache, map_location=dev, weights_only=False)
    net, feat = load_router(args.policy, cache, dev)
    pid = torch.zeros(cache["input_base"].shape[0], dtype=torch.long, device=dev)
    ah = net.logit(cache["input_base"].to(dev), None, pid).view(-1).cpu().numpy()
    A = (cache["loss_base"].view(-1) - cache["loss_super"].view(-1)).cpu().numpy()
    im_files = cache["im_file"]
    n = len(im_files)
    print(f"[*] {n} frames, feat={feat}")

    # join labels by stem -> attributes + GT boxes
    lab = {Path(x["name"]).stem: x for x in json.loads(Path(args.labels_json).read_text())}

    obj_count = np.zeros(n); small_frac = np.zeros(n); mean_area = np.zeros(n)
    is_night = np.zeros(n); is_highway = np.zeros(n)
    lumin = np.zeros(n); edge = np.zeros(n)
    for i, f in enumerate(im_files):
        stem = Path(f).stem
        rec = lab.get(stem, {})
        attrs = rec.get("attributes", {})
        is_night[i] = 1.0 if attrs.get("timeofday") == "night" else 0.0
        is_highway[i] = 1.0 if attrs.get("scene") == "highway" else 0.0
        boxes = [b for b in rec.get("labels", []) if b.get("category") not in EXCLUDE and "box2d" in b]
        areas = np.array([(b["box2d"]["x2"] - b["box2d"]["x1"]) * (b["box2d"]["y2"] - b["box2d"]["y1"])
                          for b in boxes]) if boxes else np.array([])
        obj_count[i] = len(areas)
        small_frac[i] = float((areas < SMALL_AREA).mean()) if len(areas) else 0.0
        mean_area[i] = float(areas.mean()) if len(areas) else 0.0
        img = cv2.imread(f)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lumin[i] = float(gray.mean())
            edge[i] = float((cv2.Canny(gray, 100, 200) > 0).mean())
        if (i + 1) % 2000 == 0:
            print(f"[*] processed {i+1}/{n}", flush=True)

    scalars = {
        "movable-object count": obj_count,
        "small-object fraction": small_frac,
        "mean object area": mean_area,
        "luminance": lumin,
        "edge density": edge,
    }

    # ---- correlation table: how well A-hat / A track each interpretable scalar ----
    print("\n=== |correlation| with predicted advantage A-hat  vs  true advantage A ===")
    print(f"{'scalar':22s} {'rho(Ahat)':>10s} {'rho(A)':>8s} {'r(Ahat)':>9s} {'r(A)':>8s}")
    rows = []
    for name, v in scalars.items():
        rho_ah, rho_a = spearman(ah, v), spearman(A, v)
        r_ah, r_a = pearson(ah, v), pearson(A, v)
        rows.append((name, rho_ah, rho_a, r_ah, r_a))
        print(f"{name:22s} {rho_ah:>10.3f} {rho_a:>8.3f} {r_ah:>9.3f} {r_a:>8.3f}")
    print(f"\nrho(Ahat, A) = {spearman(ah, A):.3f}   r(Ahat, A) = {pearson(ah, A):.3f}")

    # ---- decile table: bottom (D1) vs top (D10) absolute means ----
    order = np.argsort(ah)
    bins = np.array_split(order, args.nbins)
    d1, d10 = bins[0], bins[-1]
    print(f"\n=== bottom decile D1 (low A-hat) vs top decile D10 (high A-hat), n/bin~{len(d1)} ===")
    print(f"{'attribute':22s} {'D1':>9s} {'D10':>9s}")
    for name, v in {**scalars, "true advantage A": A, "night fraction": is_night,
                    "highway fraction": is_highway}.items():
        print(f"{name:22s} {v[d1].mean():>9.3f} {v[d10].mean():>9.3f}")

    # ---- figure: z-scored attribute trend across deciles ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--"})
    styles = [("movable-object count", "tab:blue", "o"),
              ("small-object fraction", "tab:red", "s"),
              ("luminance", "tab:green", "^"),
              ("edge density", "tab:orange", "D")]
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    x = np.arange(1, args.nbins + 1)
    for name, c, mk in styles:
        v = scalars[name]
        z = (v - v.mean()) / (v.std() + 1e-8)
        ax.plot(x, [z[b].mean() for b in bins], marker=mk, color=c, label=name,
                markersize=4.5, linewidth=1.6)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"predicted advantage $\hat{A}$ (decile)", fontsize=10)
    ax.set_ylabel("normalized attribute (z-score)", fontsize=10)
    ax.set_xticks(x); ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = Path(args.out_dir) / "fig_router_behavior_bdd.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"\n[*] -> {out}")


if __name__ == "__main__":
    main()
