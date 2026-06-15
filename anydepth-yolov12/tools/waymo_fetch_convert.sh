#!/bin/bash
# Stream Waymo Open Dataset v2 (parquet) one segment at a time: download the
# camera_image + camera_box parquet for a segment, convert FRONT camera -> YOLO
# images/labels, then delete the parquet. Peak disk stays ~one segment (~150 MB),
# so the full val set converts in <10 GB of working space even though the raw
# parquet total is ~30 GB.
#
# Prereq (run yourself, interactively):
#   1) Accept the Waymo license at waymo.com/open  (one-time, per Google account)
#   2) gcloud auth login        # or: gcloud auth application-default login
#
# Usage:
#   tools/waymo_fetch_convert.sh <split> [N] [emit_img_labels]
#     <split>           training | validation
#     N                 number of segments to fetch (0 = all)
#     emit_img_labels   "1" to also write per-frame YOLO txt (for router training)
#
#   tools/waymo_fetch_convert.sh validation 5         # smoke: 5 val clips
#   tools/waymo_fetch_convert.sh validation 0         # all 202 val clips
#   tools/waymo_fetch_convert.sh training 0 1         # all train + image labels
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)
# Prefer the gcloud-bundled gsutil so it shares `gcloud auth login` credentials
# (~/.config/gcloud) rather than the standalone gsutil's ~/.boto.
[ -d "$HOME/google-cloud-sdk/bin" ] && export PATH="$HOME/google-cloud-sdk/bin:$PATH"

SPLIT=${1:?split: training|validation}
N=${2:-0}
EMIT=${3:-0}
BUCKET="gs://waymo_open_dataset_v_2_0_0/${SPLIT}"
# our out_root uses val/ for validation to match the rest of the repo
OUT_SPLIT=$([ "$SPLIT" = "validation" ] && echo val || echo train)
ROOT="${WAYMO_OUT_ROOT:-/media/data/waymo}"
OUT="${ROOT}/${OUT_SPLIT}"
TMP="${ROOT}/_parquet"
mkdir -p "$OUT" "$TMP"

echo "[*] listing $BUCKET/camera_image ..."
mapfile -t SEGS < <(gsutil ls "$BUCKET/camera_image/*.parquet" | sed 's#.*/##; s#\.parquet$##')
echo "[*] $SPLIT: ${#SEGS[@]} segments available"
[ "$N" -gt 0 ] && SEGS=("${SEGS[@]:0:$N}")
echo "[*] fetching ${#SEGS[@]} segments -> $OUT"

EMIT_FLAG=""; [ "$EMIT" = "1" ] && EMIT_FLAG="--emit_img_labels"
i=0
for seg in "${SEGS[@]}"; do
  i=$((i+1))
  if [ -f "$OUT/labels/$seg.txt" ]; then echo "[$i/${#SEGS[@]}] $seg (cached, skip)"; continue; fi
  echo "[$i/${#SEGS[@]}] $seg"
  gsutil -q cp "$BUCKET/camera_image/$seg.parquet" "$TMP/img.parquet"
  gsutil -q cp "$BUCKET/camera_box/$seg.parquet"   "$TMP/box.parquet" || : >"$TMP/box.parquet"
  python tools/waymo_v2_to_yolo.py \
      --image_parquet "$TMP/img.parquet" --box_parquet "$TMP/box.parquet" \
      --out_root "$OUT" --segment "$seg" $EMIT_FLAG
  rm -f "$TMP/img.parquet" "$TMP/box.parquet"
done
echo "[*] done: $OUT  (segments=$(ls "$OUT/labels" | wc -l))"
