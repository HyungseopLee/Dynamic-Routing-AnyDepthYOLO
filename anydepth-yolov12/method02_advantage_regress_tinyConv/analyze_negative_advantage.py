"""Negative-advantage analysis: does the router detect frames where the deep (SUPER)
path actually HURTS (true advantage A = loss_base - loss_super < 0), and what kind of
scenes are those?

This probes that the router is not merely a "difficulty" proxy: it must output a LOW
(ideally negative) advantage exactly when extra depth does not help, so the threshold
test routes those frames to BASE. We quantify:
  (1) base rate of A<=0 frames,
  (2) ranking quality: AUC of using -Ahat to retrieve A<0 frames (1.0 = perfect),
      and Spearman rho restricted to the A<0 regime,
  (3) sign agreement at the operating boundary (Ahat threshold set to the A<=0 base
      rate, i.e. route the same fraction to BASE that truly should go to BASE),
  (4) for BDD, how A<0 frames differ from A>0 frames in interpretable attributes.

    conda run -n yolov12 python -m method02_advantage_regress_tinyConv.analyze_negative_advantage \
        --dataset bdd100k \
        --cache method02_advantage_regress_tinyConv/outputs/bdd100k/cache_val_g2.pt \
        --policy method02_advantage_regress_tinyConv/outputs/bdd100k/policy_scenario_s0.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from method02_advantage_regress_tinyConv.eval_video_bdd import load_policy

SMALL_AREA = 32 * 32


def auc(score, label):
    """AUC of `score` for retrieving label==1 (Mann-Whitney)."""
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bdd100k")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--labels_json", default="/media/data/bdd100k_yolo/bdd100k_labels_images_val.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    cache = torch.load(args.cache, map_location=dev, weights_only=False)
    A = (cache["loss_base"].view(-1) - cache["loss_super"].view(-1)).cpu().numpy()
    n = len(A)

    net, ckpt, feat, is_gap = load_policy(args.policy, dev)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    inp = cache["input_base"]
    if is_gap:
        inp = inp.mean(dim=(2, 3))
    pid = torch.zeros(n, dtype=torch.long, device=dev)
    with torch.no_grad():
        ahat = net.logit(inp, cache.get("pred_base"), pid).view(-1).cpu().numpy()

    neg = A <= 0.0
    base_rate = float(neg.mean())
    print(f"\n[{args.dataset}]  N={n}")
    print(f"  true advantage A: mean={A.mean():+.4f}  median={np.median(A):+.4f}  "
          f"min={A.min():+.4f}  max={A.max():+.4f}")
    print(f"  frames where SUPER hurts/ties (A<=0): {neg.sum()} ({base_rate*100:.1f}%)")
    print(f"  strictly A<0: {(A<0).sum()} ({(A<0).mean()*100:.1f}%)")

    # (2) ranking: can -Ahat retrieve A<0 frames?
    a_auc = auc(-ahat, (A < 0).astype(int))
    rho_all = spearman(ahat, A)
    rho_neg = spearman(ahat[A < 0], A[A < 0])
    print(f"\n  ranking quality")
    print(f"    AUC(-Ahat retrieves A<0)        : {a_auc:.3f}")
    print(f"    Spearman(Ahat, A) all           : {rho_all:.3f}")
    print(f"    Spearman(Ahat, A) within A<0    : {rho_neg:.3f}")

    # (3) sign agreement at the base-rate-matched boundary
    thr = np.quantile(ahat, base_rate)        # route lowest base_rate fraction to BASE
    pred_base = ahat <= thr
    tp = int((pred_base & neg).sum()); fp = int((pred_base & ~neg).sum())
    fn = int((~pred_base & neg).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"\n  base-rate-matched boundary (Ahat<={thr:+.4f} -> BASE)")
    print(f"    precision (predicted-BASE that truly A<=0): {prec*100:.1f}%")
    print(f"    recall    (true A<=0 caught)              : {rec*100:.1f}%")

    # (4) what are the A<0 scenes? (BDD attributes)
    if args.dataset.startswith("bdd") and Path(args.labels_json).exists():
        import cv2
        recs = {Path(r["name"]).stem: r for r in json.load(open(args.labels_json))}
        oc = np.zeros(n); sf = np.zeros(n); ma = np.zeros(n)
        night = np.zeros(n); hwy = np.zeros(n); lum = np.zeros(n); edge = np.full(n, np.nan)
        for i, f in enumerate(cache["im_file"]):
            r = recs.get(Path(f).stem)
            if r is None:
                continue
            at = r.get("attributes", {})
            night[i] = at.get("timeofday") == "night"
            hwy[i] = at.get("scene") == "highway"
            boxes = [l["box2d"] for l in r.get("labels", []) if "box2d" in l]
            areas = np.array([(b["x2"]-b["x1"])*(b["y2"]-b["y1"]) for b in boxes])
            oc[i] = len(areas)
            sf[i] = (areas < SMALL_AREA).mean() if len(areas) else 0.0
            ma[i] = areas.mean() if len(areas) else 0.0
            img = cv2.imread(f)
            if img is not None:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lum[i] = g.mean(); edge[i] = (cv2.Canny(g, 100, 200) > 0).mean()
        attrs = {"obj count": oc, "small-obj frac": sf, "mean area": ma,
                 "night frac": night, "highway frac": hwy, "luminance": lum,
                 "edge density": edge}
        print(f"\n  scene attributes: A<0 (deep hurts) vs A>0 (deep helps)")
        print(f"    {'attribute':<16}{'A<0':>10}{'A>0':>10}{'ratio':>8}")
        m_neg = A < 0; m_pos = A > 0
        for k, v in attrs.items():
            v = np.asarray(v, float)
            mn = np.nanmean(v[m_neg]); mp = np.nanmean(v[m_pos])
            ratio = mn / mp if mp else float("nan")
            print(f"    {k:<16}{mn:>10.3f}{mp:>10.3f}{ratio:>8.2f}")


if __name__ == "__main__":
    main()
