"""Build a YOLO image-detection *view* of the converted Waymo segments so the
anydepth detector can be finetuned at Waymo's native resolution.

The converter writes one frame-indexed label file per segment
    <root>/<split>/labels/<segment>.txt   lines: "<frame> <cls> <xc> <yc> <w> <h>"
which is what the *video* evaluator wants, but Ultralytics image training needs
one YOLO txt per frame, found by replacing '/images/' -> '/labels/' in each image
path. We therefore explode the per-segment file into
    <root>/<split>/labels/<segment>/<frame:06d>.txt   lines: "<cls> <xc> <yc> <w> <h>"
(a per-segment *directory*, which coexists with the per-segment *file*), and emit
an absolute image-path list <root>/<split>_det.txt that the data yaml points to.
Frames with no boxes get an empty label file and are still listed (true negatives).

    python tools/build_waymo_det.py --root /media/data/waymo_yolo --split val
    python tools/build_waymo_det.py --root /media/data/waymo_yolo --split train
"""
import argparse
from collections import defaultdict
from pathlib import Path


def build_split(root: Path, split: str):
    sdir = root / split
    img_root = sdir / "images"
    lab_root = sdir / "labels"
    seg_files = sorted(lab_root.glob("*.txt"))
    if not seg_files:
        print(f"[!] {split}: no per-segment label files under {lab_root}")
        return
    list_path = root / f"{split}_det.txt"
    n_img = n_seg = n_box = 0
    with open(list_path, "w") as flist:
        for segf in seg_files:
            seg = segf.stem
            seg_img_dir = img_root / seg
            if not seg_img_dir.is_dir():
                continue
            # group boxes by frame
            byframe = defaultdict(list)
            for ln in segf.read_text().splitlines():
                p = ln.split()
                if len(p) != 6:
                    continue
                fi = int(p[0])
                byframe[fi].append(" ".join(p[1:]))
                n_box += 1
            out_lab_dir = lab_root / seg
            out_lab_dir.mkdir(exist_ok=True)
            for jpg in sorted(seg_img_dir.glob("*.jpg")):
                fi = int(jpg.stem)
                (out_lab_dir / f"{fi:06d}.txt").write_text(
                    "\n".join(byframe.get(fi, [])) + ("\n" if byframe.get(fi) else ""))
                flist.write(str(jpg.resolve()) + "\n")
                n_img += 1
            n_seg += 1
    print(f"[*] {split}: {n_seg} segments, {n_img} frames, {n_box} boxes "
          f"-> {list_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/data/waymo_yolo")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    args = ap.parse_args()
    build_split(Path(args.root), args.split)


if __name__ == "__main__":
    main()
