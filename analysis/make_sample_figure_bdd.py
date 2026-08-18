"""Motivation Figure (our setting): several scenes that look hard to a human, yet the
BASE path matches or beats SUPER -- so semantic appearance does not predict where the
deep path pays off. Each panel is a busy/cluttered scene with n_base >= n_super.
Boxes: SUPER = solid blue, BASE = dashed red. Counts are detections above a fixed
visualization confidence.

    python -m analysis.make_sample_figure_bdd \
        --weight results/step1_finetune/weights/bdd100k/<alpha0.2>.pt --scan 1500
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa
from step3_eval.eval_video import BDD_MOT_EVAL_CLS  # noqa

OUT = Path(__file__).resolve().parent.parent / "outputs/figures"


def boxes_at(r, conf, cls_keep):
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4))
    xy = r.boxes.xyxy.cpu().numpy()
    sc = r.boxes.conf.cpu().numpy()
    cl = r.boxes.cls.cpu().numpy().astype(int)
    keep = (sc >= conf) & np.isin(cl, list(cls_keep))
    return xy[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--img_root", default="/media/data/bdd100k_yolo/val/images")
    ap.add_argument("--labels_json",
                    default="/media/data/bdd100k_yolo/bdd100k_labels_images_val.json")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.40, help="visualization conf")
    ap.add_argument("--scan", type=int, default=1500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--npanel", type=int, default=3, help="number of mismatch panels")
    ap.add_argument("--picks", default=None, help="comma-separated stems to force")
    ap.add_argument("--out", default=str(OUT / "fig_sample_routing.pdf"))
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    tod = {Path(m["name"]).stem: m.get("attributes", {}).get("timeofday")
           for m in json.loads(Path(args.labels_json).read_text())}
    files = sorted(Path(args.img_root).glob("*.jpg"))[:args.scan]
    yolo = YOLO(args.weight, task="detect")
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip = {"base": [True] * N, "super": [False] * N}

    def run(bgr):
        out = {}
        for tag in ("super", "base"):
            r = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=0.001,
                             iou=0.7, skip=skip[tag], verbose=False, device=dev)[0]
            out[tag] = boxes_at(r, args.conf, BDD_MOT_EVAL_CLS)
        return out

    cand = {}  # stem -> (n_super, n_base, boxes, path)
    for i, f in enumerate(files):
        b = run(cv2.imread(str(f)))
        cand[f.stem] = (len(b["super"]), len(b["base"]), b, str(f))
        if (i + 1) % 100 == 0:
            print(f"[*] scanned {i + 1}/{len(files)}", flush=True)

    # Mismatch panels: scene looks HARD (busy/cluttered, many objects) yet BASE
    # matches or beats SUPER (n_base >= n_super). Rank by how much base wins, then
    # by busyness; keep only clearly busy scenes.
    def mismatch_score(k):
        ns, nb, *_ = cand[k]
        if nb != ns or nb < 8:          # base must TIE super AND scene be busy
            return -1
        return nb                       # prefer the busiest tied scenes

    if args.picks:
        picks = args.picks.split(",")
    else:
        ranked = sorted([k for k in cand if mismatch_score(k) > 0],
                        key=mismatch_score, reverse=True)
        picks = ranked[:args.npanel]
    for k in picks:
        ns, nb, *_ = cand[k]
        print(f"[mismatch] {k} ({tod.get(k)}): super={ns} base={nb}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})

    npan = len(picks)
    fig, axes = plt.subplots(1, npan, figsize=(3.6 * npan, 2.6))
    if npan == 1:
        axes = [axes]
    for ax, stem in zip(axes, picks):
        ns, nb, boxes, fp = cand[stem]
        ax.imshow(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB))
        for x1, y1, x2, y2 in boxes["super"]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="tab:blue", lw=1.3, zorder=3))
        for x1, y1, x2, y2 in boxes["base"]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="tab:red", lw=1.1, ls=(0, (3, 2)), zorder=4))
        ax.set_title(r"Looks hard $\to$ depth wasted", fontsize=9.5)
        ax.set_xlabel(f"SUPER: {ns} det   |   BASE: {nb} det", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([0], [0], color="tab:blue", lw=1.5, label="SUPER"),
               Line2D([0], [0], color="tab:red", lw=1.5, ls=(0, (3, 2)), label="BASE")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"[*] -> {args.out}")


if __name__ == "__main__":
    main()
