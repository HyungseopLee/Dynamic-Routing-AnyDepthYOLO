"""Convert BDD100K det_20 (Scalabel JSON) -> Ultralytics YOLO format.

Reads (read-only) the original dataset on /media/bigdata and writes ONLY to
the output dir (default /media/data/bdd100k_yolo). Images are symlinked by
default (no 5GB copy, source untouched); pass --copy-images to copy instead.

Output layout (matches ultralytics/cfg/datasets/bdd100k.yaml):
    <out>/train/images/<name>.jpg   (symlink or copy)
    <out>/train/labels/<name>.txt   (YOLO: cls cx cy w h, normalized)
    <out>/val/images, <out>/val/labels

10-class mapping (det_20 category -> yaml index). Categories not listed
(other person / other vehicle / trailer) are dropped.

Usage:
    python tools/bdd_det20_to_yolo.py \
        --img_root /media/bigdata/bdd100k/bdd100k/images/100k \
        --label_dir /media/bigdata/bdd100k/bdd100k/labels/det_20 \
        --out /media/data/bdd100k_yolo
"""

import argparse
import json
import os
from pathlib import Path

# BDD100K images are uniformly 1280x720.
IMG_W, IMG_H = 1280, 720

CLASS_MAP = {
    "pedestrian": 0, "rider": 1, "car": 2, "bus": 3, "truck": 4,
    "bicycle": 5, "motorcycle": 6, "traffic light": 7, "traffic sign": 8,
    "train": 9,
}  # other person / other vehicle / trailer -> dropped


def convert_split(json_path, img_src_dir, out_dir, split, copy_images):
    print(f"[{split}] loading {json_path} ...")
    records = json.load(open(json_path))
    img_out = out_dir / split / "images"
    lbl_out = out_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_img = n_box = n_drop = n_missing = 0
    for r in records:
        name = r["name"]                       # e.g. b1c66a42-6f7d68ca.jpg
        stem = Path(name).stem
        src = Path(img_src_dir) / split / name
        if not src.exists():
            n_missing += 1
            continue

        # image link/copy
        dst = img_out / name
        if not dst.exists():
            if copy_images:
                import shutil
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)

        # labels
        lines = []
        for lab in (r.get("labels") or []):
            cat = lab.get("category")
            if cat not in CLASS_MAP or "box2d" not in lab:
                n_drop += 1
                continue
            b = lab["box2d"]
            cx = (b["x1"] + b["x2"]) / 2 / IMG_W
            cy = (b["y1"] + b["y2"]) / 2 / IMG_H
            w = (b["x2"] - b["x1"]) / IMG_W
            h = (b["y2"] - b["y1"]) / IMG_H
            lines.append(f"{CLASS_MAP[cat]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_box += 1
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
        n_img += 1

    print(f"[{split}] images={n_img} boxes={n_box} dropped_cats={n_drop} "
          f"missing_imgs={n_missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", default="/media/bigdata/bdd100k/bdd100k/images/100k")
    ap.add_argument("--label_dir", default="/media/bigdata/bdd100k/bdd100k/labels/det_20")
    ap.add_argument("--out", default="/media/data/bdd100k_yolo")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--copy-images", action="store_true",
                    help="copy images instead of symlinking (uses ~5GB+)")
    args = ap.parse_args()

    out = Path(args.out)
    for split in args.splits:
        convert_split(Path(args.label_dir) / f"det_{split}.json",
                      args.img_root, out, split, args.copy_images)
    print(f"[*] done -> {out}")


if __name__ == "__main__":
    main()
