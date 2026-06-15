"""Characterise WHEN super pays off: correlate the per-image advantage
A = loss_base - loss_super (high A => super helps) with interpretable image
properties on the BDD100K val cache.

Covariates (per image):
  n_gt           # ground-truth boxes
  mean_area      mean GT box area as fraction of the image
  med_area       median GT box area (fraction)
  frac_small     fraction of GT boxes with area-fraction < SMALL_THR
  n_small        count of such small boxes
  mean_side_px   mean sqrt(area) in pixels (object scale)
  lum            mean luminance (Y)            [pixel signal, subsampled]
  edge           mean Sobel gradient magnitude [pixel signal, subsampled]

Reports Pearson + Spearman of each covariate vs A, plus a top-decile vs
bottom-decile-A contrast table. Saves a json + a scatter/という figure.

    python method02_advantage_regress_tinyConv/analyze_lossgap.py \
        --cache outputs/bdd100k/cache_val_both.pt --img_root /media/data/bdd100k_yolo/val
"""
import argparse, json
from pathlib import Path
import numpy as np, torch

SMALL_THR = 0.001   # area-fraction; ~ <(0.032*W x 0.032*H) box


def load_labels(p):
    if not Path(p).exists():
        return np.zeros((0, 4))
    rows = []
    for ln in Path(p).read_text().splitlines():
        f = ln.split()
        if len(f) >= 5:
            rows.append([float(f[1]), float(f[2]), float(f[3]), float(f[4])])
    return np.array(rows) if rows else np.zeros((0, 4))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="method02_advantage_regress_tinyConv/outputs/bdd100k/cache_val_both.pt")
    ap.add_argument("--img_root", default="/media/data/bdd100k_yolo/val")
    ap.add_argument("--pix_n", type=int, default=3000, help="images to read for lum/edge (0=skip)")
    ap.add_argument("--out", default="method02_advantage_regress_tinyConv/outputs/bdd100k/eval/lossgap_bdd.json")
    args = ap.parse_args()

    c = torch.load(args.cache, map_location="cpu", weights_only=False)
    A = (c["loss_base"].view(-1) - c["loss_super"].view(-1)).numpy()
    files = list(c["im_file"]); n = len(files)
    lbl_dir = Path(args.img_root) / "labels"

    n_gt = np.zeros(n); mean_area = np.zeros(n); med_area = np.zeros(n)
    frac_small = np.zeros(n); n_small = np.zeros(n); mean_side = np.zeros(n)
    H = W = None
    import cv2
    for i, f in enumerate(files):
        b = load_labels(lbl_dir / f"{Path(f).stem}.txt")
        n_gt[i] = len(b)
        if len(b):
            area = b[:, 2] * b[:, 3]            # fraction of image
            mean_area[i] = area.mean(); med_area[i] = np.median(area)
            frac_small[i] = float((area < SMALL_THR).mean())
            n_small[i] = int((area < SMALL_THR).sum())
            mean_side[i] = float(np.sqrt(area).mean())

    # pixel signals on a subsample (image reads are the slow part)
    lum = np.full(n, np.nan); edge = np.full(n, np.nan)
    if args.pix_n:
        idx = np.linspace(0, n - 1, min(args.pix_n, n)).astype(int)
        for i in idx:
            im = cv2.imread(files[i])
            if im is None:
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            lum[i] = g.mean()
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
            edge[i] = np.sqrt(gx * gx + gy * gy).mean()

    covs = {"n_gt": n_gt, "mean_area": mean_area, "med_area": med_area,
            "frac_small": frac_small, "n_small": n_small, "mean_side": mean_side,
            "lum": lum, "edge": edge}

    print(f"[*] n={n}  A: mean={A.mean():.4f} std={A.std():.4f} "
          f"(>0: super helps, frac>0={np.mean(A>0):.3f})")
    print(f"\n{'covariate':12s} {'pearson':>9s} {'spearman':>9s}   (vs A = loss_base-loss_super)")
    corr = {}
    for k, v in covs.items():
        m = np.isfinite(v) & np.isfinite(A)
        if m.sum() < 30 or np.std(v[m]) == 0:
            continue
        p = float(np.corrcoef(v[m], A[m])[0, 1]); s = spearman(v[m], A[m])
        corr[k] = {"pearson": p, "spearman": s, "n": int(m.sum())}
        print(f"{k:12s} {p:9.3f} {s:9.3f}")

    # top vs bottom decile of A: what do the high-gain images look like?
    q = np.quantile(A, [0.1, 0.9])
    lowm, highm = A <= q[0], A >= q[1]
    print(f"\n--- high-A decile (super helps most) vs low-A decile (super hurts/neutral) ---")
    print(f"{'covariate':12s} {'low-A':>10s} {'high-A':>10s} {'ratio':>8s}")
    contrast = {}
    for k, v in covs.items():
        lo = np.nanmean(v[lowm]); hi = np.nanmean(v[highm])
        contrast[k] = {"low": float(lo), "high": float(hi)}
        r = hi / lo if lo else float("nan")
        print(f"{k:12s} {lo:10.4f} {hi:10.4f} {r:8.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n": n, "A_mean": float(A.mean()), "A_std": float(A.std()),
         "frac_super_helps": float(np.mean(A > 0)),
         "corr": corr, "decile_contrast": contrast, "small_thr": SMALL_THR}, indent=2))
    print(f"\n[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
