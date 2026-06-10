"""Quick probe: does running BASE at LOW resolution widen the base->super
headroom and per-image advantage variance (the routing signal) on BDD?

Compares, per image, AP(base@low-res, skip-all) vs AP(super@full-res, no-skip):
reports dataset mAP gap and the std of the per-image AP advantage. Bigger std =
more routable heterogeneity. Run in yolov12 env.

    python method02_advantage_regress_tinyConv/probe_resroute.py \
        --weight <alpha0.2.pt> --base_imgsz 360 640 --super_imgsz 720 1280 --limit 1500
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa
import eval_baseline_kitti as B  # noqa
from method02_advantage_regress_tinyConv.eval_video_bdd import BDD_MOT_EVAL_CLS  # noqa
from method02_advantage_regress_tinyConv.eval_image_protocol_bdd import load_gt  # noqa


def img_ap(preds, gts):
    m, gtc = B.match_frame_multi_iou(preds, gts)
    _, map50, _ = B.dataset_map_multi_iou(m, gtc)
    return map50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--img_root", default="/media/data/bdd100k_yolo/val")
    ap.add_argument("--base_imgsz", type=int, nargs=2, default=[360, 640])
    ap.add_argument("--super_imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    B.EVAL_CLS = BDD_MOT_EVAL_CLS
    yolo = YOLO(args.weight, task="detect")
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip_base = [True] * N
    skip_super = [False] * N

    img_dir = Path(args.img_root) / "images"; lbl_dir = Path(args.img_root) / "labels"
    imgs = sorted(img_dir.glob("*.jpg"))[:args.limit]

    advs = []
    mb, ms = [defaultdict(list), defaultdict(int)], [defaultdict(list), defaultdict(int)]
    for i, fp in enumerate(imgs):
        bgr = cv2.imread(str(fp)); H, W = bgr.shape[:2]
        gts = load_gt(lbl_dir / f"{fp.stem}.txt", W, H)
        rb = yolo.predict(source=bgr, imgsz=tuple(args.base_imgsz), conf=args.conf, iou=0.7,
                          skip=skip_base, verbose=False, device=args.device)[0]
        rs = yolo.predict(source=bgr, imgsz=tuple(args.super_imgsz), conf=args.conf, iou=0.7,
                          skip=skip_super, verbose=False, device=args.device)[0]
        pb, ps = B.boxes_to_preds(rb), B.boxes_to_preds(rs)
        advs.append(img_ap(ps, gts) - img_ap(pb, gts))           # per-image AP advantage
        for store, preds in ((mb, pb), (ms, ps)):
            m, gtc = B.match_frame_multi_iou(preds, gts)
            store[0]["all"].extend(m)
            for c, n in gtc.items():
                store[1][c] += n
        if (i + 1) % 300 == 0:
            print(f"[*] {i+1}/{len(imgs)}")

    def ds_map(store):
        m = store[0]["all"]; _, mp, _ = B.dataset_map_multi_iou(m, store[1]); return mp
    ap_base, ap_super = ds_map(mb), ds_map(ms)
    advs = np.array(advs)
    print(f"\n=== res-routing probe ({len(imgs)} imgs) ===")
    print(f"base @ {tuple(args.base_imgsz)} (skip-all)  mAP50 = {ap_base:.4f}")
    print(f"super@ {tuple(args.super_imgsz)} (no-skip)  mAP50 = {ap_super:.4f}")
    print(f"dataset headroom (super-base)             = {ap_super-ap_base:+.4f}")
    print(f"per-image AP-advantage  std = {advs.std():.4f}  |median|={np.median(np.abs(advs)):.4f}")
    print(f"  frac(super strictly helps) = {(advs>0).mean():.3f}")
    print("  (compare: current layer-skip base@full  headroom +0.0060, AP-adv std 0.0422)")


if __name__ == "__main__":
    main()
