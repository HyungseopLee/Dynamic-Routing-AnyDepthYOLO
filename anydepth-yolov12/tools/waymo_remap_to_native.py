"""One-time, lossless migration of the converted Waymo labels from the transient
BDD class-id encoding (used by waymo_v2_to_yolo.py during download) to Waymo's
own native taxonomy.

During download we stored Waymo box types under BDD ids (WAYMO_TO_BDD = {1:2, 2:0,
4:1, 3:8}). Now we want Waymo-native ids: 0 vehicle, 1 pedestrian, 2 cyclist.
(Waymo 2D camera boxes never contain SIGN, so it is dropped entirely.) The map
below is the bijection BDD-id -> native-id; running it once over every per-segment
labels/<seg>.txt makes the stored data match ultralytics/cfg/datasets/waymo.yaml.
A .native_classes sentinel per split guards against double-application (the id
ranges overlap, so a second pass would corrupt).

    python tools/waymo_remap_to_native.py --root /media/data/waymo_yolo --split val
    python tools/waymo_remap_to_native.py --root /media/data/waymo_yolo --split train
"""
import argparse
from pathlib import Path

# stored BDD id -> Waymo native id        (Waymo type -> BDD was {1:2,2:0,4:1,3:8})
REMAP = {2: 0,   # VEHICLE
         0: 1,   # PEDESTRIAN
         1: 2}   # CYCLIST  (BDD-8 SIGN is absent in Waymo 2D and is dropped)


def remap_split(root: Path, split: str):
    sdir = root / split
    lab_root = sdir / "labels"
    sentinel = sdir / ".native_classes"
    if sentinel.exists():
        print(f"[=] {split}: already native ({sentinel}), skip")
        return
    seg_files = sorted(lab_root.glob("*.txt"))
    if not seg_files:
        print(f"[!] {split}: no label files under {lab_root}")
        return
    n_seg = n_box = 0
    bad = set()
    for segf in seg_files:
        out = []
        for ln in segf.read_text().splitlines():
            p = ln.split()
            if len(p) != 6:
                continue
            old = int(p[1])
            if old not in REMAP:
                bad.add(old)
                continue
            p[1] = str(REMAP[old])
            out.append(" ".join(p))
            n_box += 1
        segf.write_text("\n".join(out) + ("\n" if out else ""))
        n_seg += 1
    if bad:
        print(f"[!] {split}: unexpected ids dropped {sorted(bad)} "
              f"(not in BDD-id set {sorted(REMAP)})")
    sentinel.write_text("labels remapped BDD-id -> Waymo native "
                        "{0:vehicle,1:pedestrian,2:sign,3:cyclist}\n")
    print(f"[*] {split}: remapped {n_seg} segments, {n_box} boxes -> native")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/data/waymo_yolo")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    args = ap.parse_args()
    remap_split(Path(args.root), args.split)


if __name__ == "__main__":
    main()
