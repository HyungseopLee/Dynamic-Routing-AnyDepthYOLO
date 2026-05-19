"""
Video Inference Pipeline for AnyDepth-YOLOv12 on KITTI Tracking Dataset

Evaluates model on consecutive video frames and reports:
- mean mAP (AP@50, AP@50:95)
- mean FPS
- mean Latency (ms)
- mean Energy (mJ, GPU)

Usage:
    python eval_video_kitti.py \
        --weight ./runs/bdd100k/detect/anydepth-yolov12l/.../weights/best.pt \
        --kitti_root /media/data/kitti-tracking \
        --sequences 0000 0001 0002 \
        --imgsz 192 640 \
        --conf 0.5 \
        --project ./runs/kitti-tracking/video_eval
"""

import os
import sys
import json
import time
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO


# =============================================================================
# KITTI <-> BDD100K class mapping
# =============================================================================
# KITTI type -> BDD100K class index
# BDD100K: {0:person, 1:rider, 2:car, 3:bus, 4:truck, 5:bike, 6:motor, 7:traffic light, 8:traffic sign, 9:train}
KITTI_TO_BDD100K = {
    'Car': 2,
    'Van': 2,        # Van -> car
    'Truck': 4,
    'Pedestrian': 0,
    'Person_sitting': 0,  # Person_sitting -> person
    'Cyclist': 1,    # Cyclist -> rider
    'Tram': 9,       # Tram -> train
}
# Ignored: DontCare, Misc

BDD100K_NAMES = {0: 'person', 1: 'rider', 2: 'car', 3: 'bus', 4: 'truck',
                 5: 'bike', 6: 'motor', 7: 'traffic light', 8: 'traffic sign', 9: 'train'}

# Only evaluate on classes that exist in both datasets
EVAL_CLASSES = sorted(set(KITTI_TO_BDD100K.values()))  # [0, 1, 2, 4, 9]


# =============================================================================
# KITTI Label Parser
# =============================================================================
def parse_kitti_labels(label_path):
    """Parse KITTI tracking label file and group by frame.

    Returns:
        dict: {frame_idx: [(class_id, x1, y1, x2, y2), ...]}
    """
    gt_by_frame = defaultdict(list)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            frame_idx = int(parts[0])
            obj_type = parts[2]

            if obj_type not in KITTI_TO_BDD100K:
                continue

            class_id = KITTI_TO_BDD100K[obj_type]
            x1 = float(parts[6])
            y1 = float(parts[7])
            x2 = float(parts[8])
            y2 = float(parts[9])

            gt_by_frame[frame_idx].append((class_id, x1, y1, x2, y2))

    return gt_by_frame


# =============================================================================
# mAP Calculation
# =============================================================================
def compute_iou(box1, box2):
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_ap(precisions, recalls):
    """Compute AP using 101-point interpolation (COCO-style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = mpre[mrec >= t]
        ap += (p.max() if len(p) > 0 else 0.0) / 101

    return ap


def evaluate_frame(preds, gts, iou_threshold=0.5):
    """Match predictions to ground truths for a single frame.

    Args:
        preds: list of (class_id, conf, x1, y1, x2, y2)
        gts: list of (class_id, x1, y1, x2, y2)
        iou_threshold: IoU threshold for matching

    Returns:
        list of (class_id, conf, is_tp) for each prediction
        int: number of ground truths per class
    """
    results = []
    gt_count = defaultdict(int)
    gt_matched = [False] * len(gts)

    for gt_cls, *_ in gts:
        if gt_cls in EVAL_CLASSES:
            gt_count[gt_cls] += 1

    # Sort predictions by confidence (descending)
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

    for pred_cls, conf, px1, py1, px2, py2 in preds_sorted:
        if pred_cls not in EVAL_CLASSES:
            continue

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, (gt_cls, gx1, gy1, gx2, gy2) in enumerate(gts):
            if gt_cls != pred_cls or gt_matched[gt_idx]:
                continue
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
        if is_tp:
            gt_matched[best_gt_idx] = True

        results.append((pred_cls, conf, is_tp))

    return results, dict(gt_count)


def compute_map(all_results, all_gt_counts, iou_thresholds=None):
    """Compute mAP across all frames.

    Args:
        all_results: list of (class_id, conf, is_tp) across all frames
        all_gt_counts: dict {class_id: total_gt_count}
        iou_thresholds: list of IoU thresholds. None = [0.5]

    Returns:
        dict with AP per class and mAP
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    ap_per_class = {}

    for cls_id in EVAL_CLASSES:
        cls_results = [(conf, tp) for c, conf, tp in all_results if c == cls_id]
        n_gt = all_gt_counts.get(cls_id, 0)

        if n_gt == 0:
            continue

        # Sort by confidence descending
        cls_results.sort(key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for conf, is_tp in cls_results:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            recalls.append(tp_cumsum / n_gt)

        ap = compute_ap(np.array(precisions), np.array(recalls))
        ap_per_class[cls_id] = ap

    mean_ap = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    return ap_per_class, float(mean_ap)


# =============================================================================
# Energy Monitoring (GPU)
# =============================================================================
class EnergyMonitor:
    """GPU energy monitor using pynvml."""

    def __init__(self):
        self.available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
            self.available = True
        except Exception:
            print("[!] pynvml not available. Energy measurement disabled.")

    def get_power_mw(self):
        """Get current GPU power draw in milliwatts."""
        if not self.available:
            return 0.0
        return self.pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts

    def compute_energy_mj(self, power_mw, duration_s):
        """Compute energy in millijoules: E = P * t."""
        return power_mw * duration_s  # mW * s = mJ


# =============================================================================
# Main Pipeline
# =============================================================================
def run_sequence(model, seq_id, kitti_root, args, energy_monitor):
    """Run inference on a single KITTI tracking sequence.

    Returns:
        dict with per-frame and aggregated metrics
    """
    img_dir = Path(kitti_root) / 'training' / 'image_02' / seq_id
    label_path = Path(kitti_root) / 'training' / 'label_02' / f'{seq_id}.txt'

    if not img_dir.exists():
        print(f"[!] Image directory not found: {img_dir}")
        return None
    if not label_path.exists():
        print(f"[!] Label file not found: {label_path}")
        return None

    # Parse GT
    gt_by_frame = parse_kitti_labels(label_path)

    # Get sorted frame list
    frame_files = sorted(img_dir.glob('*.png'))
    if not frame_files:
        frame_files = sorted(img_dir.glob('*.jpg'))
    n_frames = len(frame_files)
    print(f"\n[*] Sequence {seq_id}: {n_frames} frames, {len(gt_by_frame)} frames with GT")

    # Use skip config from args
    skip = args.skip_list

    # Prepare output video
    sample_img = cv2.imread(str(frame_files[0]))
    h, w = sample_img.shape[:2]

    out_dir = Path(args.project) / f'seq_{seq_id}'
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / 'per_frame_predictions'
    pred_dir.mkdir(exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(str(out_dir / 'output.mp4'), fourcc, 10, (w, h))

    # Per-frame logging
    latency_log = []
    energy_log = []
    all_match_results = []
    all_gt_counts = defaultdict(int)

    warmup_frames = min(10, n_frames)

    for idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        frame_idx = int(frame_path.stem)

        # Measure power before inference
        power_before = energy_monitor.get_power_mw()

        # Inference
        predict_kwargs = dict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )
        if skip is not None:
            predict_kwargs['skip'] = skip

        results = model.predict(**predict_kwargs)
        result = results[0]

        # Measure power after inference
        power_after = energy_monitor.get_power_mw()
        avg_power_mw = (power_before + power_after) / 2.0

        # Timing
        latency_ms = result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']
        latency_s = latency_ms / 1000.0
        energy_mj = energy_monitor.compute_energy_mj(avg_power_mw, latency_s)

        # Skip warmup frames for metrics
        is_warmup = idx < warmup_frames
        if not is_warmup:
            latency_log.append(latency_ms)
            energy_log.append(energy_mj)

        # Extract predictions: (class_id, conf, x1, y1, x2, y2)
        preds = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                preds.append((cls_id, conf, x1, y1, x2, y2))

        # Match with GT (for mAP)
        gts = gt_by_frame.get(frame_idx, [])

        # AP@50
        match_results_50, gt_count_50 = evaluate_frame(preds, gts, iou_threshold=0.5)
        all_match_results.append(('50', match_results_50, gt_count_50))
        for cls_id, cnt in gt_count_50.items():
            all_gt_counts[cls_id] += cnt

        # Save per-frame predictions
        frame_pred = {
            'frame_idx': frame_idx,
            'latency_ms': round(latency_ms, 2),
            'energy_mJ': round(energy_mj, 2),
            'n_detections': len(preds),
            'n_gt': len(gts),
            'detections': [
                {'class': BDD100K_NAMES.get(p[0], str(p[0])), 'conf': round(p[1], 4),
                 'bbox': [round(v, 1) for v in p[2:]]}
                for p in preds
            ],
        }
        with open(pred_dir / f'frame_{frame_idx:06d}.json', 'w') as f:
            json.dump(frame_pred, f, indent=2)

        # Draw boxes on frame
        annotated = draw_boxes(frame, preds, gts)
        video_out.write(annotated)

        # Progress
        if (idx + 1) % 50 == 0 or idx == n_frames - 1:
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0
            print(f"  [{idx+1}/{n_frames}] latency={latency_ms:.1f}ms  fps={fps:.1f}  "
                  f"det={len(preds)}  gt={len(gts)}")

    video_out.release()

    # Aggregate match results for mAP
    all_matches_flat = []
    for _, matches, _ in all_match_results:
        all_matches_flat.extend(matches)

    # AP@50
    ap_per_class_50, mean_ap_50 = compute_map(all_matches_flat, dict(all_gt_counts), [0.5])

    # AP@50:95 (compute at multiple IoU thresholds)
    # Re-evaluate at multiple thresholds requires re-matching, simplified here
    # For accurate AP@50:95, we'd need to store raw preds/gts and re-match
    # Using AP@50 as primary metric for now

    # Compute summary (excluding warmup)
    mean_latency = np.mean(latency_log) if latency_log else 0.0
    mean_fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    mean_energy = np.mean(energy_log) if energy_log else 0.0

    summary = {
        'sequence': seq_id,
        'total_frames': n_frames,
        'warmup_frames': warmup_frames,
        'eval_frames': n_frames - warmup_frames,
        'mean_AP50': round(float(mean_ap_50), 4),
        'AP50_per_class': {BDD100K_NAMES.get(k, str(k)): round(v, 4) for k, v in ap_per_class_50.items()},
        'mean_FPS': round(float(mean_fps), 2),
        'mean_latency_ms': round(float(mean_latency), 2),
        'mean_energy_mJ': round(float(mean_energy), 2),
        'model_config': {
            'is_anydepth': getattr(model.model, 'num_skippable_layers', 0) > 0,
            'num_skippable_layers': getattr(model.model, 'num_skippable_layers', 0),
            'skip': ''.join('T' if s else 'F' for s in skip) if skip else 'N/A',
            'imgsz': args.imgsz if isinstance(args.imgsz, int) else list(args.imgsz),
            'conf_threshold': args.conf,
        }
    }

    # Save summary
    with open(out_dir / 'summary_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save latency/energy CSV
    with open(out_dir / 'latency_energy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'latency_ms', 'energy_mJ'])
        for i, (lat, eng) in enumerate(zip(latency_log, energy_log)):
            writer.writerow([i + warmup_frames, round(lat, 2), round(eng, 2)])

    return summary


def draw_boxes(frame, preds, gts):
    """Draw prediction (green) and GT (red) boxes on frame."""
    annotated = frame.copy()

    # GT boxes (red)
    for cls_id, x1, y1, x2, y2 in gts:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        label = BDD100K_NAMES.get(cls_id, str(cls_id))
        cv2.putText(annotated, f'GT:{label}', (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Prediction boxes (green)
    for cls_id, conf, x1, y1, x2, y2 in preds:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f'{BDD100K_NAMES.get(cls_id, str(cls_id))} {conf:.2f}'
        cv2.putText(annotated, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return annotated


def main():
    parser = argparse.ArgumentParser(description='Video inference pipeline on KITTI Tracking')
    parser.add_argument('--weight', type=str, required=True, help='Path to model weights')
    parser.add_argument('--kitti_root', type=str, default='/media/data/kitti-tracking',
                        help='KITTI tracking dataset root')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Sequence IDs to evaluate (e.g., 0000 0001 0002). '
                             'If not specified, all sequences in kitti_root are used.')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640],
                        help='Input image size. Single value for square (640), '
                             'or two values for rect (h w), e.g., --imgsz 192 640 for KITTI, '
                             '--imgsz 384 640 for BDD100K. Values should be multiples of 32.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--skip', type=str, default=None,
                        help='Skip config for each skippable layer. '
                             'e.g., --skip FFTTFFTT (F=False/use layer, T=True/skip layer). '
                             'Length must match num_skippable_layers. '
                             'If not specified: Full mode (all F). '
                             'Ignored for baseline models.')
    parser.add_argument('--task', type=str, default='detect', help='Task type')
    parser.add_argument('--project', type=str, default='./runs/kitti-tracking/video_eval',
                        help='Output project directory')
    args = parser.parse_args()

    # Parse imgsz: single int → square, two ints → (h, w)
    if len(args.imgsz) == 1:
        args.imgsz = args.imgsz[0]  # e.g., 640 → square
    elif len(args.imgsz) == 2:
        args.imgsz = tuple(args.imgsz)  # e.g., (192, 640) → rect
    else:
        raise ValueError(f'--imgsz expects 1 or 2 values, got {len(args.imgsz)}')

    # Auto-detect all sequences if not specified
    if args.sequences is None:
        label_dir = Path(args.kitti_root) / 'training' / 'label_02'
        args.sequences = sorted([f.stem for f in label_dir.glob('*.txt')])
        print(f'[*] Auto-detected {len(args.sequences)} sequences: {args.sequences}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.project = os.path.join(args.project, timestamp)
    os.makedirs(args.project, exist_ok=True)

    # Load model
    print(f'[*] Loading model: {args.weight}')
    model = YOLO(args.weight, task=args.task)
    num_skip = getattr(model.model, 'num_skippable_layers', 0)
    is_anydepth = num_skip > 0

    # Parse skip config
    if is_anydepth:
        if args.skip is None:
            args.skip_list = [False] * num_skip  # Default: Full mode
        else:
            if len(args.skip) != num_skip:
                raise ValueError(f'--skip length ({len(args.skip)}) != num_skippable_layers ({num_skip}). '
                                 f'Expected {num_skip} characters of T/F, e.g., {"F"*num_skip}')
            args.skip_list = [c == 'T' for c in args.skip.upper()]
        skip_str = ''.join('T' if s else 'F' for s in args.skip_list)
    else:
        args.skip_list = None
        skip_str = 'N/A (baseline)'

    print(f'[*] AnyDepth: {is_anydepth}, num_skippable_layers: {num_skip}')
    print(f'[*] Skip config: {skip_str}')
    print(f'[*] Image size: {args.imgsz}, Conf threshold: {args.conf}')
    print(f'[*] Sequences: {args.sequences}')
    print(f'[*] Output: {args.project}')

    # Energy monitor
    energy_monitor = EnergyMonitor()

    # Run each sequence
    all_summaries = []
    for seq_id in args.sequences:
        summary = run_sequence(model, seq_id, args.kitti_root, args, energy_monitor)
        if summary:
            all_summaries.append(summary)
            print(f'\n  >> Seq {seq_id}: mAP@50={summary["mean_AP50"]:.4f}  '
                  f'FPS={summary["mean_FPS"]:.1f}  '
                  f'Latency={summary["mean_latency_ms"]:.1f}ms  '
                  f'Energy={summary["mean_energy_mJ"]:.1f}mJ')

    # Overall summary across all sequences
    if all_summaries:
        overall = {
            'timestamp': timestamp,
            'weight': args.weight,
            'task': args.task,
            'imgsz': args.imgsz,
            'conf': args.conf,
            'is_anydepth': is_anydepth,
            'num_skippable_layers': num_skip,
            'sequences': args.sequences,
            'per_sequence': all_summaries,
            'overall': {
                'mean_AP50': round(np.mean([s['mean_AP50'] for s in all_summaries]), 4),
                'mean_FPS': round(np.mean([s['mean_FPS'] for s in all_summaries]), 2),
                'mean_latency_ms': round(np.mean([s['mean_latency_ms'] for s in all_summaries]), 2),
                'mean_energy_mJ': round(np.mean([s['mean_energy_mJ'] for s in all_summaries]), 2),
                'total_frames': sum(s['total_frames'] for s in all_summaries),
            }
        }

        with open(os.path.join(args.project, 'overall_summary.json'), 'w') as f:
            json.dump(overall, f, indent=2)

        print(f'\n{"="*60}')
        print(f'Overall Results ({len(all_summaries)} sequences, '
              f'{overall["overall"]["total_frames"]} frames):')
        print(f'  mean AP@50:     {overall["overall"]["mean_AP50"]:.4f}')
        print(f'  mean FPS:       {overall["overall"]["mean_FPS"]:.1f}')
        print(f'  mean Latency:   {overall["overall"]["mean_latency_ms"]:.1f} ms')
        print(f'  mean Energy:    {overall["overall"]["mean_energy_mJ"]:.1f} mJ')
        print(f'{"="*60}')
        print(f'Results saved to: {args.project}')


if __name__ == '__main__':
    main()


'''
--sequences 0000 0001 \


# Baseline (KITTI: 1242 x 375 → imgsz 192 640)
mkdir -p ./runs/kitti/tracking/baseline-yolov12l
python eval_video_kitti.py \
        --weight ./finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.6_orig_mAP34.3_33.1.pt \
        --kitti_root /media/data/kitti-tracking \
        --imgsz 192 640 \
        --conf 0.5 \
        --project ./runs/kitti/tracking/baseline-yolov12l \
        2>&1 | tee ./runs/kitti/tracking/baseline-yolov12l/eval.log


# AnyDepth Full (all layers used, skip=FFFFFFFF)
mkdir -p ./runs/kitti/tracking/anydepth-yolov12l-full
python eval_video_kitti.py \
        --weight ./finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.6_orig_mAP34.3_33.1.pt \
        --kitti_root /media/data/kitti-tracking \
        --imgsz 192 640 \
        --conf 0.5 \
        --skip FFFFFFFF \
        --project ./runs/kitti/tracking/anydepth-yolov12l-full \
        2>&1 | tee ./runs/kitti/tracking/anydepth-yolov12l-full/eval.log


# AnyDepth Base (all layers skipped, skip=TTTTTTTT)
mkdir -p ./runs/kitti/tracking/anydepth-yolov12l-base
python eval_video_kitti.py \
        --weight ./finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.6_orig_mAP34.3_33.1.pt \
        --kitti_root /media/data/kitti-tracking \
        --imgsz 192 640 \
        --conf 0.5 \
        --skip TTTTTTTT \
        --project ./runs/kitti/tracking/anydepth-yolov12l-base \
        2>&1 | tee ./runs/kitti/tracking/anydepth-yolov12l-base/eval.log


'''
