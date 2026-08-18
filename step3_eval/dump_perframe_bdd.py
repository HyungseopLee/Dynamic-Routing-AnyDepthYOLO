"""Per-frame routing dump for BDD100K MOT, grouped by sequence and tagged with the
scene condition (time-of-day / scene / weather from the detection attributes, which
share the MOT video IDs). Feeds the realistic condition-drift budget-tracking
experiment (make_scenario_figures_bdd.py): sequences are concatenated in a chosen
condition order so the PI controller is exercised under night<->dawn, city<->highway,
clear<->rainy drift rather than abrupt KITTI boundaries.

Per labeled (5 fps) frame we store exactly what the simulator replays:
  ahat_base / ahat_super : router advantage when the previous path was base / super
  match_base / match_super : AP-match records of each path vs GT (poolable)
  gt_count : per-class GT count (path-independent)

    python -m step3_eval.dump_perframe_bdd \
        --weight results/step1_finetune/weights/bdd100k/<alpha0.2>.pt \
        --router results/step2_router/weights/bdd100k/router_scenario_s0.pt \
        --grid 2 --imgsz 720 1280 --sequences seqA,seqB,...
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO  # noqa

from step3_eval import eval_baseline_kitti as B  # noqa
from router.feature_tap import INPUT_LEVEL_LAYERS, STATE_LAYERS  # noqa
from step3_eval.eval_video import (  # noqa
    parse_box_track, labeled_frames, load_router, grid_vec, BDD_MOT_EVAL_CLS)

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--mot_root", default="/media/data/bdd100k_mot/val")
    ap.add_argument("--labels_json",
                    default="/media/data/bdd100k_yolo/bdd100k_labels_images_val.json")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--sequences", default="", help="comma-separated seq ids (required)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT / "bdd100k/perframe_scenarios.pkl"))
    args = ap.parse_args()

    B.EVAL_CLS = BDD_MOT_EVAL_CLS
    dev = args.device if torch.cuda.is_available() else "cpu"
    cond = {x["name"].split(".")[0]: x.get("attributes", {})
            for x in json.loads(Path(args.labels_json).read_text())}
    seqs = [s for s in args.sequences.split(",") if s]
    if not seqs:
        raise SystemExit("--sequences is required")

    yolo = YOLO(args.weight, task="detect")
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super, skip_base = [False] * N, [True] * N
    captured = {}
    for idx in STATE_LAYERS:
        yolo.model.model[idx].register_forward_hook(
            lambda m, i, o, k=idx: captured.__setitem__(k, o))

    net, ckpt, feat, is_gap = load_router(args.router, dev)
    one = torch.ones(1, dtype=torch.long, device=dev)
    zero = torch.zeros(1, dtype=torch.long, device=dev)

    label_dir = Path(args.mot_root) / "labels"
    video_dir = Path(args.mot_root) / "videos"

    out = {"seqs": {}, "meta": {"weight": args.weight, "router": args.router,
                                "grid": args.grid, "imgsz": args.imgsz}}
    for si, seq in enumerate(seqs):
        gt = parse_box_track(label_dir / f"{seq}.json")
        frames = []
        for fidx, bgr in labeled_frames(video_dir / f"{seq}.mov", gt):
            captured.clear()
            r_s = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                               iou=0.7, skip=skip_super, verbose=False, device=dev)[0]
            in_s = grid_vec(captured, INPUT_LEVEL_LAYERS, args.grid).unsqueeze(0)
            captured.clear()
            r_b = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                               iou=0.7, skip=skip_base, verbose=False, device=dev)[0]
            in_b = grid_vec(captured, INPUT_LEVEL_LAYERS, args.grid).unsqueeze(0)
            xs = in_s.mean(dim=(2, 3)) if is_gap else in_s
            xb = in_b.mean(dim=(2, 3)) if is_gap else in_b
            with torch.no_grad():
                ahat_super = float(net.logit(xs, None, one))
                ahat_base = float(net.logit(xb, None, zero))
            gts = gt.get(fidx, [])
            m_s, gtc = B.match_frame_multi_iou(B.boxes_to_preds(r_s), gts)
            m_b, _ = B.match_frame_multi_iou(B.boxes_to_preds(r_b), gts)
            frames.append({"ahat_base": ahat_base, "ahat_super": ahat_super,
                           "match_base": m_b, "match_super": m_s,
                           "gt_count": dict(gtc)})
        out["seqs"][seq] = {"cond": cond.get(seq, {}), "frames": frames}
        print(f"[{si+1}/{len(seqs)} {seq}] {len(frames)} frames  cond={cond.get(seq,{})}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(args.out, "wb"))
    print(f"[*] -> {args.out}")


if __name__ == "__main__":
    main()
