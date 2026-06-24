"""Dump per-frame routing material for the latency-budget scheduling experiment.

The closed-loop controller (sim_latency_budget.py) needs, for every video frame,
the two quantities that a deployed router would have access to plus the ground
truth needed to score whichever path it keeps:

  per frame (in temporal order, grouped by sequence):
    - ahat_base  : router advantage when the PREVIOUS executed path was BASE
    - ahat_super : router advantage when the PREVIOUS executed path was SUPER
    - match_base / match_super : AP-match records (cls, conf, [tp@IoU]) of each
                                 path's detections vs. GT (poolable over any frame
                                 subset to recompute AP under a dynamic schedule)
    - gt_count   : per-class GT count for the frame (path-independent)

Because per-frame cost is memoryless, the simulator can replay ANY tau schedule
offline from this dump without re-running the detector. Latency/energy per path
are taken from the measured RTX-3090 anchors at sim time (deterministic), so the
detector forward only needs to happen once here.

    python -m method_advantage_regress.eval.dump_perframe_kitti \
        --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
        --policy method_advantage_regress/outputs/kitti/ablation/policy_input_g2_s0.pt \
        --grid 2 --imgsz 384 1248 --conf 0.001
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from ultralytics import YOLO

import eval_baseline_kitti as B
from method_advantage_regress.router.feature_tap import INPUT_LEVEL_LAYERS, STATE_LAYERS
from method_advantage_regress.eval.eval_video import load_policy, grid_vec
from method_advantage_regress.train.build_cache import num_skippable

BASE = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt")
    ap.add_argument("--policy", default=str(BASE / "kitti/ablation/policy_input_g2_s0.pt"))
    ap.add_argument("--kitti_root", default="/media/data/kitti-tracking")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1248])
    ap.add_argument("--grid", default="2")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--out", default=str(BASE / "kitti/perframe_dump.pkl"))
    args = ap.parse_args()
    G = (int(args.grid), int(args.grid))
    device = args.device if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.weight, task="detect")
    yolo.model.to(device).eval()
    yolo.model.num_skippable_layers = num_skippable(yolo.model)
    N = yolo.model.num_skippable_layers
    skip_super, skip_base = [False] * N, [True] * N

    net, _ = load_policy(args.policy, device)
    in_c = (768 // len(INPUT_LEVEL_LAYERS)) * len(INPUT_LEVEL_LAYERS)
    with torch.no_grad():
        net(torch.zeros(2, in_c, G[0], G[1], device=device), None,
            torch.zeros(2, dtype=torch.long, device=device))
    net.load_state_dict(torch.load(args.policy, map_location=device, weights_only=False)["state_dict"])
    net.eval()

    captured = {}
    for idx in STATE_LAYERS:
        yolo.model.model[idx].register_forward_hook(
            lambda m, i, o, k=idx: captured.__setitem__(k, o))

    label_dir = Path(args.kitti_root) / "training" / "label_02"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.txt"))

    # warmup
    warm = np.zeros((args.imgsz[0], args.imgsz[1], 3), dtype=np.uint8)
    for sk in (skip_super, skip_base):
        for _ in range(3):
            yolo.predict(source=warm, imgsz=tuple(args.imgsz), conf=args.conf,
                         iou=0.7, skip=sk, verbose=False, device=device)

    one = torch.ones(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, dtype=torch.long, device=device)
    frames_out = []
    for seq in seqs:
        img_dir = Path(args.kitti_root) / "training" / "image_02" / seq
        lpath = label_dir / f"{seq}.txt"
        if not img_dir.exists() or not lpath.exists():
            print(f"[!] skip {seq}"); continue
        gt_by_frame, dc_by_frame = B.parse_kitti_labels(lpath)
        frames = sorted(img_dir.glob("*.png")) or sorted(img_dir.glob("*.jpg"))
        if args.limit > 0:
            frames = frames[: args.limit]
        for fi, fpath in enumerate(frames):
            frame_idx = int(fpath.stem)
            bgr = cv2.imread(str(fpath))
            if bgr is None:
                continue
            captured.clear()
            r_super = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                                   iou=0.7, skip=skip_super, verbose=False, device=device)[0]
            in_s = grid_vec(captured, INPUT_LEVEL_LAYERS, G).unsqueeze(0)
            captured.clear()
            r_base = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                                  iou=0.7, skip=skip_base, verbose=False, device=device)[0]
            in_b = grid_vec(captured, INPUT_LEVEL_LAYERS, G).unsqueeze(0)
            with torch.no_grad():
                ahat_super = float(net.logit(in_s, None, one).view(-1))
                ahat_base = float(net.logit(in_b, None, zero).view(-1))

            gts = gt_by_frame.get(frame_idx, [])
            dc = dc_by_frame.get(frame_idx, [])
            rec = {"seq": seq, "first": fi == 0,
                   "ahat_base": ahat_base, "ahat_super": ahat_super}
            for tag, r in (("base", r_base), ("super", r_super)):
                p = B.boxes_to_preds(r)
                p = B.filter_dontcare(p, dc) if dc else p
                m, gtc = B.match_frame_multi_iou(p, gts)
                rec[f"match_{tag}"] = m
                rec["gt_count"] = gtc  # same for both paths
            frames_out.append(rec)
        print(f"[seq {seq}] cumulative {len(frames_out)} frames")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"meta": {"weight": args.weight, "policy": args.policy,
                              "grid": G, "imgsz": args.imgsz, "conf": args.conf},
                     "frames": frames_out}, f)
    print(f"[*] dumped {len(frames_out)} frames -> {out}")


if __name__ == "__main__":
    main()
