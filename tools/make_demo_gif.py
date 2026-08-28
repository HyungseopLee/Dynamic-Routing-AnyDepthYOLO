"""Compress a slice of a demo video into an animated GIF for the README teaser.

The demo mp4s are ~400 MB each -- far too large for git. This extracts a short
window, downscales it, and writes an optimised GIF small enough to commit.

Uses only OpenCV + Pillow (no ffmpeg required).

Usage (run from repo root):
    python -m tools.make_demo_gif \
        --video results/step4_deploy/demo_videos/demo_video_night_dawn_fps199-292.mp4 \
        --out   docs/demo_teaser.gif \
        --start 60 --dur 12 --width 640 --fps 10
"""
import argparse
from pathlib import Path

import cv2
from PIL import Image


def extract(video, start_s, dur_s, width, out_fps):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / out_fps))
    first = int(start_s * src_fps)
    last = first + int(dur_s * src_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames, idx = [], first
    while idx < last:
        ok, bgr = cap.read()
        if not ok:
            break
        if (idx - first) % step == 0:
            h, w = bgr.shape[:2]
            scale = width / w
            small = cv2.resize(bgr, (width, int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    return frames, src_fps / step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--dur", type=float, default=12.0, help="duration (s)")
    ap.add_argument("--width", type=int, default=640, help="output width (px)")
    ap.add_argument("--fps", type=float, default=10.0, help="output GIF fps")
    ap.add_argument("--colors", type=int, default=128,
                    help="palette size; lower shrinks the file")
    args = ap.parse_args()

    frames, fps = extract(args.video, args.start, args.dur, args.width, args.fps)
    if not frames:
        raise SystemExit("no frames extracted -- check --start/--dur")

    # A single adaptive palette for the whole clip keeps colours stable between frames and
    # compresses far better than per-frame palettes -- but it must be sampled across the WHOLE
    # clip, not just frame 0. A night->day clip palettised from its first (night) frame has no
    # daylight colours left and turns the sky cyan. Tile a strip of evenly spaced frames and
    # quantize that instead, so every lighting condition contributes.
    sample = frames[:: max(1, len(frames) // 24)][:24]
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(sample)))
    for i, f in enumerate(sample):
        strip.paste(f, (0, frames[0].height * i))
    pal = strip.quantize(colors=args.colors, method=Image.MEDIANCUT)
    quantized = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in frames]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(out, save_all=True, append_images=quantized[1:],
                      duration=int(round(1000.0 / fps)), loop=0, optimize=True)
    mb = out.stat().st_size / 1e6
    print(f"[*] {out}  {len(frames)} frames  {frames[0].size[0]}x{frames[0].size[1]}"
          f"  {fps:.1f} fps  {mb:.1f} MB")


if __name__ == "__main__":
    main()
