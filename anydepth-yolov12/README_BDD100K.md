# BDD100K Setup for the Depth-Routing Policy

## Pipeline Overview

The detector is a frozen AnyDepth-YOLOv12 with two inference paths — a cheap
**BASE** path and an expensive **SUPER** path. A small **policy network** decides,
per frame, which path to run. The goal is to keep detection accuracy near SUPER
while spending far fewer FLOPs by sending only the hard frames to SUPER.

Two stages, two different data modalities:

1. **Train the policy — on detection *images*.** The frozen detector is run once
   per image on both paths; we cache the pooled features and the per-image
   detection loss of each path. The policy is trained to predict which path is
   worth it (the BASE−SUPER loss *advantage*). Training is per-image because the
   signal (does SUPER help on *this* input?) is per-image and needs no temporal
   context.

2. **Evaluate the policy — on *video*.** Routing is only meaningful over a
   sequence: the decision for frame *t* is made recursively from frame *t−1*'s
   chosen-path state. So we need labeled, contiguous video to measure the
   accuracy/FLOPs trade-off of the learned routing — not isolated images.

That split is exactly why we need two different parts of BDD100K below.

---

## Why MOT (`box_track_20`) for Evaluation

BDD100K ships 100,000 raw 40s videos, but **the raw videos have no box labels**
(only GPS/IMU) — you cannot measure detection AP on them. Per-frame box labels
exist only for the **MOT (multi-object tracking) subset**, `box_track_20`:

- 2,000 of the 100k videos are selected and labeled end-to-end at **5 fps**.
- Splits: **train 1,400 / val 200 / test 400** (test labels are withheld).
- This is the *only* part of BDD100K that gives labeled, contiguous video, so it
  is the only valid target for evaluating a temporal routing policy with AP.

**Use the val split (200 videos).** The 1,400 train videos are drawn from the
same pool as the detection keyframes used to train the policy, so evaluating on
them risks train/eval leakage. Val is clean: the policy never saw it.

The detection images used for training come from a different bundle (`det_20`):
one representative keyframe per video, ~70k train / 10k val stills with box
labels. (BDD100K releases labels per task, so detection stills and MOT video are
separate downloads covering different subsets of the same 100k videos.)

| Bundle | Location (under `$BDD_SRC`) | What it is | Use |
|---|---|---|---|
| Raw video | `video_parts/*.zip` | ~100k `.mov`, no box labels | (decoded for eval frames) |
| detection `det_20` | `images/100k/...` + `labels/det_20/det_{train,val}.json` | 70k/10k labeled stills | policy training |
| MOT `box_track_20` | `labels/box_track_20/{train,val}/*.json` | 1,400/200 labeled videos @5fps | policy evaluation |

---

## How the Video Labels Are Structured

One json **per video** (`labels/box_track_20/val/<video>.json`). It is a list of
**frame** entries; each frame is one 5fps sample of the 40s clip:

```jsonc
[
  {
    "name": "<video>-0000001.jpg",   // frame file name (1-based)
    "videoName": "<video>",
    "frameIndex": 0,                 // 0-based position in the 5fps sequence
    "labels": [
      { "id": "00122062",            // track id (same object across frames)
        "category": "car",           // one of the MOT classes
        "box2d": { "x1":.., "y1":.., "x2":.., "y2":.. } }  // pixels, 1280×720
    ]
  },
  ...   // ~200 frames per video (frameIndex 0..~201)
]
```

- Coordinates are absolute pixels in a **1280×720** image.
- MOT categories observed: `car, truck, bus, pedestrian, rider, motorcycle`
  (the full MOT vocabulary also allows `bicycle, train`).
- `track id` links the same object across frames — useful for MOT metrics, **not
  needed for detection AP** (which is per-frame).

---

## How to Measure Detection AP on the Video

The labels are sparse (5fps) but routing must run densely (the sequence drives
the recursive decision). So:

1. **Decode every frame** of each `.mov` with `cv2.VideoCapture` (~30fps, ~1,196
   frames per clip). Run the policy + chosen detector path on **every** frame so
   the temporal routing state is correct.
2. **Score AP only on the labeled frames.** The labeled frames are a 5fps subset;
   map them to decoded frame indices by
   `video_frame ≈ round(frameIndex * fps / 5)` (≈ every 6th frame). Accumulate
   matches/GT only on those frames.
3. **Map classes** between the detector and the MOT labels. The detector predicts
   the 10-class detection taxonomy; MOT labels use the movable-object subset.
   Score on the common classes:
   `pedestrian, rider, car, truck, bus, motorcycle, bicycle, train`
   (drop `traffic light` / `traffic sign`, which MOT does not annotate).
4. **Check the frame orientation.** `cv2` may report a clip as 720×1280
   (portrait) due to rotation metadata, while `box2d` is in 1280×720 — verify
   `frame.shape` in the loop and transpose/align if needed before matching.

Scale note: MOT val is ~200 videos × ~200 labeled frames ≈ **~40k scored frames**
at 1280×720, and every decoded frame runs both detector paths during evaluation —
substantially heavier than a small image set. Start with a subset
(`--sequences` / `--limit`) before a full run.

---

## Data Preparation

Treat the original download as read-only; write all outputs elsewhere.
Below, `$BDD_SRC` is the read-only original download and `$BDD_OUT` is a writable
output root.
ㄱ
### 1. Detection → YOLO format (for policy training)

`det_20` is Scalabel JSON (13 categories); ultralytics expects normalized YOLO txt.

```bash
cd anydepth-yolov12
python tools/bdd_det20_to_yolo.py \
    --img_root  "$BDD_SRC/images/100k" \
    --label_dir "$BDD_SRC/labels/det_20" \
    --out       "$BDD_OUT/bdd100k_yolo"
```

10-class mapping (det_20 category → `bdd100k.yaml` index):

```
pedestrian→0  rider→1  car→2  bus→3  truck→4
bicycle→5  motorcycle→6  traffic light→7  traffic sign→8  train→9
(other person / other vehicle / trailer → dropped)
```

All images are 1280×720, so
`cx=(x1+x2)/2/1280, cy=(y1+y2)/2/720, w=(x2-x1)/1280, h=(y2-y1)/720`.

Output `$BDD_OUT/bdd100k_yolo/` (point `path:` in `bdd100k.yaml` here):
```
train/images + train/labels (txt)   # 1,273,707 boxes
val/images   + val/labels   (txt)   #   185,945 boxes
```

### Label provenance & re-preparation

The YOLO detection labels were **re-generated from the official `det_20` labels**
and replace an earlier set converted from the Kaggle `bdd100k-yolo` mirror.

**Why the change.** The Kaggle mirror is slightly incomplete: it has 367 fewer
boxes on val (185,578 vs the official **185,945**) and does not exactly match the
official 10-class taxonomy. The official `det_20` conversion is the correct
reference and is what the detector mAP numbers are measured against.

There are two ways to obtain the correct labels:

**A. Regenerate from source (canonical).** If you have the original `det_20`
Scalabel JSON on your machine, just run the converter — this is the source of
truth and needs no downloads:

```bash
python tools/bdd_det20_to_yolo.py \
    --img_root  "$BDD_SRC/images/100k" \
    --label_dir "$BDD_SRC/labels/det_20" \
    --out       "$BDD_OUT/bdd100k_yolo"
```

**B. Download the converted labels (shortcut).** On servers that do **not** have
the `det_20` JSON (only the YOLO dataset), grab the prepared label tarball
instead — it carries `train/labels` + `val/labels` only (no images, ~tens of MB):

> Drive: [`bdd100k_official_labels.tar.gz`](https://drive.google.com/file/d/1IAhUtk3F0vpfsZrMDSyshTHWE6D9OpNp/view?usp=drive_link)

```bash
cd /path/to/bdd100k_yolo

# 1) back up the existing (Kaggle) labels — do NOT delete them
mv train/labels train/labels_kaggle_bak
mv val/labels   val/labels_kaggle_bak

# 2) extract the official labels in their place
tar xzf bdd100k_official_labels.tar.gz        # restores train/labels, val/labels

# 3) invalidate the ultralytics label cache (regenerated on next run)
rm -f train/labels.cache val/labels.cache
```

> ⚠️ Removing `labels.cache` is mandatory. ultralytics caches parsed labels, so
> leaving a stale cache silently trains on the OLD labels.

To produce the tarball yourself (from a machine that already has the official
labels in place):

```bash
cd /path/to/bdd100k_yolo
tar czf bdd100k_official_labels.tar.gz train/labels val/labels
```

### 2. MOT val video (for evaluation)

All 200 val `.mov` files live in a single zip (`bdd100k_videos_val_00.zip`).
Extract only the 200 that match the labels (~3.7 GB) and copy the labels.

```bash
cd "$BDD_SRC"
DST="$BDD_OUT/bdd100k_mot"
mkdir -p "$DST/val/videos" "$DST/val/labels"

cp labels/box_track_20/val/*.json "$DST/val/labels/"

ls "$DST/val/labels" | sed 's/\.json$//' > /tmp/val_names.txt
unzip -l video_parts/bdd100k_videos_val_00.zip | awk '/\.mov$/{print $NF}' \
    | grep -Ff /tmp/val_names.txt > /tmp/extract_list.txt
unzip -j -o video_parts/bdd100k_videos_val_00.zip $(cat /tmp/extract_list.txt) \
    -d "$DST/val/videos"
```

Output `$BDD_OUT/bdd100k_mot/val/`:
```
videos/  200 .mov files (3.8 GB)
labels/  200 json files (165 MB)     # label↔video mismatch: 0
```

---

## Next Steps (TODO)

- [ ] **Policy training**: `method01_advantage_regress/build_cache.py` (bdd100k,
      `--imgsz 720 1280`) → `train_policy.py`. The cache stores GAP vectors only
      (~0.8 GB), so loss/lambda sweeps are cheap.
- [ ] **Video evaluation**: port `eval_video.py` to BDD — frame source from PNG
      directory → `.mov` decoding, label parser from KITTI txt → box_track json,
      add the class mapping and labeled-frame AP scoring described above.
