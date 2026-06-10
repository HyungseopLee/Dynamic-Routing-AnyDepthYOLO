"""Per-image (non-recursive, paper-faithful) check of the cheap routing signals.

The video eval shows luminance/edge/confidence routers falling *below random* on
BDD MOT. The paper ("Pixels Know When to Go Deep") reports the opposite on BDD val
(all match-or-exceed the oracle). This isolates whether that gap is a measurement
artifact (recursive prev-frame signal, mixed base/super conf, eval-set tau) or a
genuine detector/dataset property, by reproducing the paper's exact protocol:

  signal per image x:
    lum(x)  = Y-mean                 (Eq 7)   low -> SUPER
    edge(x) = Sobel-magnitude mean   (Eq 8)   low -> SUPER
    conf(x) = mean top-20 BASENET detection score (Eq 6)  low -> SUPER
  advantage A(x) = loss_base - loss_super   (>0: SUPER helps)

For each signal we report Pearson r vs A and, more decisively, the *oracle-direction
routing gain*: route the b-fraction of images with the LOWEST signal to SUPER and
measure the mean advantage captured vs random (which captures 0 in expectation).
gain>0 => beats random; gain<0 => below random (what the video curve shows).
"""
import argparse, sys
from pathlib import Path
import numpy as np, cv2, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa
import eval_baseline_kitti as B  # noqa
from method02_advantage_regress_tinyConv.eval_video_bdd import pixel_signals  # noqa


def routing_gain(sig, A, budgets, lower_to_super=True):
    """Mean advantage captured by routing the `b`-fraction with the most-favorable
    signal to SUPER, minus random (=b*mean(A)). Positive => signal beats random."""
    order = np.argsort(sig if lower_to_super else -sig)   # most-super-worthy first
    out = {}
    n = len(A)
    for b in budgets:
        k = int(round(b * n))
        captured = A[order[:k]].sum() / n          # mean advantage realized
        rand = b * A.mean()                        # random routes b-fraction
        out[b] = float(captured - rand)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/bdd100k/cache_val.pt")
    ap.add_argument("--weight", default="./finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.2_orig_mAP35.1_33.8.pt")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--n", type=int, default=2000, help="# val images to probe")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    c = torch.load(args.cache, map_location="cpu", weights_only=False)
    A_all = (c["loss_base"] - c["loss_super"]).numpy()
    files = c["im_file"]
    idx = np.arange(min(args.n, len(files)))

    yolo = YOLO(args.weight, task="detect")
    N = getattr(yolo.model, "num_skippable_layers", 0) or sum(
        1 for _ in []) or 8
    skip_base = [True] * N

    lum, edge, conf, A = [], [], [], []
    for j, i in enumerate(idx):
        bgr = cv2.imread(files[i])
        if bgr is None:
            continue
        l, e = pixel_signals(bgr)
        r = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf, iou=0.7,
                         skip=skip_base, verbose=False, device=args.device)[0]
        cf = r.boxes.conf.cpu() if (r.boxes is not None and len(r.boxes)) else torch.empty(0)
        lum.append(l); edge.append(e)
        conf.append(B.conf_top_k_mean(cf, 20) if cf.numel() else 0.0)
        A.append(A_all[i])
        if j % 200 == 0:
            print(f"[{j}/{len(idx)}]")
    lum, edge, conf, A = map(np.array, (lum, edge, conf, A))

    budgets = [0.1, 0.25, 0.5, 0.75]
    print(f"\n=== per-image signal vs advantage A (n={len(A)}, A mean={A.mean():.4f} std={A.std():.4f}) ===")
    print("paper expects: low signal -> SUPER, so r(signal,A) should be NEGATIVE\n")
    print(f"{'signal':>10}{'pearson_r':>11}   routing gain @ budget (lower->super)")
    for nm, s in (("luminance", lum), ("edge", edge), ("conf_top20", conf)):
        r = float(np.corrcoef(s, A)[0, 1])
        g = routing_gain(s, A, budgets, lower_to_super=True)
        gain_str = "  ".join(f"b{int(b*100)}:{g[b]:+.4f}" for b in budgets)
        print(f"{nm:>10}{r:>11.4f}   {gain_str}")
    # reference: a perfect oracle (route highest-A to super) and its inverse
    g_oracle = routing_gain(-A, A, budgets, lower_to_super=True)
    print(f"{'ORACLE(A)':>10}{1.0:>11.2f}   " +
          "  ".join(f"b{int(b*100)}:{g_oracle[b]:+.4f}" for b in budgets))
    print("\n[i] gain>0 beats random; gain<0 is below random (matches video curve).")


if __name__ == "__main__":
    main()
