"""Decisive frame/label alignment test for the MOT video eval.

For a few clips, sample labeled frames, and for each compute the SUPER-path mAP50
against that label's GT at decoded-frame offsets delta in [-DMAX, DMAX] around the
mapped position round(fi*fps/5). If alignment is correct, AP peaks at delta=0; a
peak elsewhere reveals a systematic label/frame mismatch tanking the video AP.

    python method02_advantage_regress_tinyConv/probe_align.py --weight <pt> --clips 5
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa
import eval_baseline_kitti as B  # noqa
from method02_advantage_regress_tinyConv.eval_video_bdd import (  # noqa
    BDD_MOT_EVAL_CLS, parse_box_track, open_upright, decode_upright,
    decoded_frame_for_label)


def img_map50(preds, gts):
    m, gtc = B.match_frame_multi_iou(preds, gts)
    _, mp, _ = B.dataset_map_multi_iou(m, gtc)
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--mot_root", default="/media/data/bdd100k_mot/val")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--dmax", type=int, default=3)
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--label_stride", type=int, default=10, help="use every Nth label")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    B.EVAL_CLS = BDD_MOT_EVAL_CLS
    yolo = YOLO(args.weight, task="detect")
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super = [False] * N

    lbl_dir = Path(args.mot_root) / "labels"; vid_dir = Path(args.mot_root) / "videos"
    clips = sorted(p.stem for p in lbl_dir.glob("*.json"))[:args.clips]

    # delta -> list of per-label AP
    per_delta = defaultdict(list)
    for clip in clips:
        gt = parse_box_track(lbl_dir / f"{clip}.json")
        cap, meta = open_upright(vid_dir / f"{clip}.mov")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = fps / 5.0
        labels = sorted(gt)[::args.label_stride]
        # for each chosen label, the decoded frames we need: round(fi*step)+delta
        want = {}                      # decoded_idx -> list of (fi, delta)
        for fi in labels:
            m = decoded_frame_for_label(fi, step)        # shared source-of-truth mapping
            for d in range(-args.dmax, args.dmax + 1):
                want.setdefault(m + d, []).append((fi, d))
        need_max = max(want) if want else -1
        di = 0
        while di <= need_max:
            ok, frame = cap.read()
            if not ok:
                break
            if di in want:
                bgr = decode_upright(frame, meta)
                r = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf, iou=0.7,
                                 skip=skip_super, verbose=False, device=args.device)[0]
                preds = B.boxes_to_preds(r)
                for fi, d in want[di]:
                    if gt[fi]:                                  # skip empty-GT frames
                        per_delta[d].append(img_map50(preds, gt[fi]))
            di += 1
        cap.release()
        print(f"[*] {clip}: {len(labels)} labels probed")

    print(f"\n=== alignment AP vs frame offset (clips={args.clips}) ===")
    print(f"{'delta':>6}{'mAP50':>9}{'n':>6}")
    best_d, best_ap = 0, -1
    for d in range(-args.dmax, args.dmax + 1):
        v = per_delta.get(d, [])
        ap_d = float(np.mean(v)) if v else 0.0
        if ap_d > best_ap:
            best_ap, best_d = ap_d, d
        mark = "  <= mapped (delta=0)" if d == 0 else ""
        print(f"{d:>6}{ap_d:>9.4f}{len(v):>6}{mark}")
    print(f"\n[*] AP peaks at delta={best_d} (mAP50={best_ap:.4f}); "
          f"delta=0 -> {np.mean(per_delta.get(0,[0])):.4f}")
    if best_d == 0:
        print("[*] alignment OK -> low video AP is genuine (domain), not a matching bug.")
    else:
        print(f"[!] peak at delta={best_d}: systematic {best_d}-frame label/frame offset.")


if __name__ == "__main__":
    main()
