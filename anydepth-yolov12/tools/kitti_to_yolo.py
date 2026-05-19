"""
Convert KITTI 2D Object Detection into YOLO format (7 classes).

Class mapping (KITTI label string -> YOLO class index):
    Car            -> 0
    Van            -> 1
    Truck          -> 2
    Pedestrian     -> 3
    Person_sitting -> 4
    Cyclist        -> 5
    Tram           -> 6
    Misc           -> dropped
    DontCare       -> dropped

Train/Val split:
    Default = 80:20 (5985 train / 1496 val), deterministic with --seed.
    Pass --split mv3d for the 3712/3769 Chen split (requires
    tools/kitti_mv3d_val_ids.txt), or --split none for everything in train.

Input layout (KITTI Object Detection 2D, "left color images"):
    {kitti_root}/training/image_2/{000000.png, ...}     (7481 images)
    {kitti_root}/training/label_2/{000000.txt, ...}

Output layout (YOLO):
    {out_root}/images/train/*.png
    {out_root}/images/val/*.png
    {out_root}/labels/train/*.txt   (yolo format: cls cx cy w h)
    {out_root}/labels/val/*.txt

Usage:
    python tools/kitti_to_yolo.py \
        --kitti_root /media/data/kitti \
        --out_root /media/data/kitti_yolo
    # default is --split 8020 --seed 0 (5985 train / 1496 val).
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image


CLASS_MAP = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
}
DROP_TYPES = {"Misc", "DontCare"}


# Geiger / MV3D split — val image ids (3769 ids). Train = remaining 3712 of 0..7480.
# Reference: https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/image_sets/val.txt
# We embed the val id list compactly by encoding the bool mask for ids 0..7480.
# A simpler approach: derive train/val from the well-known list.
# Here we use the deterministic split from Chen et al. (MV3D) widely re-used.
def load_mv3d_val_ids():
    """Return the canonical MV3D val image-id set (3769 ids of 0..7480)."""
    # The full list is large; we embed it via an external txt file if present,
    # otherwise fall back to an interleaved deterministic split.
    txt = Path(__file__).with_name("kitti_mv3d_val_ids.txt")
    if txt.exists():
        return set(int(x) for x in txt.read_text().split() if x.strip())
    # Fallback: pseudo-random but deterministic split (NOT the official one).
    # Train ~= ids where (id * 2654435761) % 2 == 0; gives roughly 50/50.
    # This is just a placeholder — please put the official list in kitti_mv3d_val_ids.txt.
    print("[!] kitti_mv3d_val_ids.txt not found; falling back to interleaved split (NOT official MV3D).")
    return {i for i in range(7481) if (i * 2654435761) % 7481 % 2 == 0}


def kitti_to_yolo_line(parts, img_w, img_h):
    """Convert one KITTI label line to a YOLO label line (str), or None if dropped."""
    obj_type = parts[0]
    if obj_type in DROP_TYPES:
        return None
    if obj_type not in CLASS_MAP:
        return None  # unknown class — skip
    cls = CLASS_MAP[obj_type]
    x1, y1, x2, y2 = map(float, parts[4:8])
    # clip to image
    x1 = max(0.0, min(img_w - 1, x1))
    x2 = max(0.0, min(img_w - 1, x2))
    y1 = max(0.0, min(img_h - 1, y1))
    y2 = max(0.0, min(img_h - 1, y2))
    w = x2 - x1; h = y2 - y1
    if w <= 1 or h <= 1:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = w / img_w; nh = h / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True,
                    help="KITTI root containing training/image_2 and training/label_2")
    ap.add_argument("--out_root", required=True,
                    help="output root (will create images/{train,val} and labels/{train,val})")
    ap.add_argument("--split", choices=["8020", "mv3d", "none"], default="8020")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for 80:20 split (deterministic).")
    ap.add_argument("--copy_images", action="store_true",
                    help="copy images (default: symlink to save disk)")
    args = ap.parse_args()

    kitti_root = Path(args.kitti_root)
    img_dir = kitti_root / "training" / "image_2"
    lbl_dir = kitti_root / "training" / "label_2"
    if not img_dir.exists() or not lbl_dir.exists():
        raise SystemExit(f"missing {img_dir} or {lbl_dir}")

    image_files = sorted(img_dir.glob("*.png"))
    print(f"[*] {len(image_files)} images in {img_dir}")

    if args.split == "mv3d":
        val_ids = load_mv3d_val_ids()
    elif args.split == "8020":
        import random
        rng = random.Random(args.seed)
        all_ids = list(range(len(image_files)))
        rng.shuffle(all_ids)
        n_val = round(len(all_ids) * 0.20)
        val_ids = set(all_ids[:n_val])
        print(f"[*] 80:20 split (seed={args.seed}): train={len(all_ids)-n_val}, val={n_val}")
    else:
        val_ids = set()

    if args.split == "none":
        def assign(i): return "train"
    else:
        def assign(i): return "val" if i in val_ids else "train"

    out_root = Path(args.out_root)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    n_boxes_kept = 0
    n_boxes_dropped = 0
    cls_counts = {c: 0 for c in CLASS_MAP.values()}

    for img_path in image_files:
        img_id = int(img_path.stem)
        split = assign(img_id)
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        yolo_lines = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 8:
                    continue
                converted = kitti_to_yolo_line(parts, img_w, img_h)
                if converted is None:
                    n_boxes_dropped += 1
                else:
                    yolo_lines.append(converted)
                    n_boxes_kept += 1
                    cls_counts[int(converted.split()[0])] += 1

        # write label
        out_lbl = out_root / "labels" / split / f"{img_path.stem}.txt"
        out_lbl.write_text("\n".join(yolo_lines))

        # image: symlink or copy
        out_img = out_root / "images" / split / img_path.name
        if out_img.exists() or out_img.is_symlink():
            out_img.unlink()
        if args.copy_images:
            shutil.copy2(img_path, out_img)
        else:
            out_img.symlink_to(img_path.resolve())

        if split == "train": n_train += 1
        else:                n_val   += 1

    print(f"\n[*] wrote train={n_train}, val={n_val} images to {out_root}")
    print(f"[*] kept {n_boxes_kept} boxes, dropped {n_boxes_dropped} (Misc/DontCare/degenerate)")
    name_of = {v: k for k, v in CLASS_MAP.items()}
    print("[*] class distribution:")
    for c in sorted(cls_counts):
        print(f"    {c} {name_of[c]:14s}: {cls_counts[c]}")


if __name__ == "__main__":
    main()
