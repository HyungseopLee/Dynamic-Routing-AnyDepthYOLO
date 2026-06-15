"""Convert ONE Waymo Open Dataset v2 (parquet, modular) segment into YOLO-style
images + labels, keeping only the FRONT camera. No TensorFlow / waymo_open_dataset
package needed -- reads the parquet directly with pyarrow, so it is light and
robust to the heavy TF dependency chain.

Inputs (per segment, v2.0.0 layout):
    <split>/camera_image/<segment>.parquet   # all cameras x all frames, JPEG bytes
    <split>/camera_box/<segment>.parquet      # 2D boxes (pixel coords), per camera

Outputs (our pipeline format):
    images/<segment>/<frame_idx:06d>.jpg                 # FRONT camera frames, ordered
    labels/<segment>.txt   lines: "<frame_idx> <cls> <xc> <yc> <w> <h>"  (normalized)
The per-segment label file mirrors the KITTI-tracking layout the video evaluator
already understands (one file per clip, frame-indexed), so video eval needs only a
parser swap. For router *image* training we additionally emit per-frame YOLO txt
(labels_img/<segment>__<frame>.txt) next to a flat image symlink farm if --flat.

Column names in v2 parquet are flattened (e.g. '[CameraImageComponent].image',
'key.camera_name'); we locate them by substring so minor version drift is tolerated.

Waymo 2D type -> BDD-anydepth detector class id:
    1 VEHICLE -> 2 car      2 PEDESTRIAN -> 0 person
    4 CYCLIST -> 1 rider    3 SIGN       -> 8 traffic sign
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FRONT = 1  # Waymo CameraName.FRONT
WAYMO_TO_BDD = {1: 2, 2: 0, 4: 1, 3: 8}


def find_col(cols, *needles):
    """First column whose lowercased name contains all needles (substring match)."""
    for c in cols:
        cl = c.lower()
        if all(n in cl for n in needles):
            return c
    return None


def load_boxes(box_path):
    """frame_timestamp -> list[(cls, xc, yc, w, h) in PIXELS] for the FRONT camera."""
    if not Path(box_path).exists():
        return {}
    t = pq.read_table(box_path)
    cols = t.column_names
    c_cam = find_col(cols, "camera_name")
    c_ts = find_col(cols, "frame_timestamp")
    c_type = find_col(cols, "type")
    c_cx = find_col(cols, "box", "center.x")
    c_cy = find_col(cols, "box", "center.y")
    c_sx = find_col(cols, "box", "size.x")
    c_sy = find_col(cols, "box", "size.y")
    d = t.to_pydict()
    out = {}
    for i in range(len(d[c_ts])):
        if d[c_cam][i] != FRONT:
            continue
        cls = WAYMO_TO_BDD.get(d[c_type][i])
        if cls is None:
            continue
        out.setdefault(d[c_ts][i], []).append(
            (cls, float(d[c_cx][i]), float(d[c_cy][i]), float(d[c_sx][i]), float(d[c_sy][i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_parquet", required=True)
    ap.add_argument("--box_parquet", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--segment", required=True, help="segment stem (folder/label name)")
    ap.add_argument("--emit_img_labels", action="store_true",
                    help="also write per-frame YOLO txt for router image training")
    args = ap.parse_args()

    boxes = load_boxes(args.box_parquet)

    t = pq.read_table(args.image_parquet)
    cols = t.column_names
    c_cam = find_col(cols, "camera_name")
    c_ts = find_col(cols, "frame_timestamp")
    c_img = find_col(cols, "image")
    d = t.to_pydict()

    # FRONT frames in temporal order
    fr = sorted({d[c_ts][i] for i in range(len(d[c_ts])) if d[c_cam][i] == FRONT})
    ts_to_idx = {ts: k for k, ts in enumerate(fr)}

    out = Path(args.out_root)
    img_dir = out / "images" / args.segment
    img_dir.mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    lab_img_dir = out / "labels_img" / args.segment
    if args.emit_img_labels:
        lab_img_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    W = H = None
    lab_lines = []
    n_img = 0
    for i in range(len(d[c_ts])):
        if d[c_cam][i] != FRONT:
            continue
        ts = d[c_ts][i]; fi = ts_to_idx[ts]
        arr = np.frombuffer(d[c_img][i], dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        cv2.imwrite(str(img_dir / f"{fi:06d}.jpg"), im)
        n_img += 1
        rows = boxes.get(ts, [])
        per_frame = []
        for cls, cx, cy, w, h in rows:
            xc, yc, ww, hh = cx / W, cy / H, w / W, h / H
            lab_lines.append(f"{fi} {cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            per_frame.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
        if args.emit_img_labels:
            (lab_img_dir / f"{fi:06d}.txt").write_text("\n".join(per_frame) + ("\n" if per_frame else ""))

    (out / "labels" / f"{args.segment}.txt").write_text("\n".join(lab_lines) + ("\n" if lab_lines else ""))
    print(f"[*] {args.segment}: {n_img} FRONT frames ({W}x{H}), {len(lab_lines)} boxes")


if __name__ == "__main__":
    main()
