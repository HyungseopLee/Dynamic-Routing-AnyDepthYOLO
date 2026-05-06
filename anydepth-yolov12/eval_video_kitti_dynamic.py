"""
Dynamic-routing video inference on KITTI Tracking.

Routing policies:
    --routing-policy prev   (default)  skip_t from router(frame_{t-1}),
                                       router OFF critical path (real-time video).
                                       Frame 0 seeded via --init-route {super,base}.
    --routing-policy same              skip_t from router(frame_t),
                                       router ON critical path (DynamicDet image-style).

For each frame:
    1) determine skip config per the chosen policy
    2) run detector with the skip config
    3) router forward (before or after detector depending on policy)

Reports per-sequence and overall:
    - SUPER / BASE ratio
    - mean AP@50
    - mean GFLOPs (SUPER×r_super + BASE×r_base + router, fixed router cost)
    - mean FPS, mean Latency, mean Energy (mJ)

Supports multiple thresholds in a single run (--thres-file or --thres).

Usage (prev-frame routing, real-time deployment):
    python eval_video_kitti_dynamic.py \
        --weight runs/.../step2_router/weights/last.pt \
        --kitti_root /media/data/kitti-tracking \
        --imgsz 192 640 --conf 0.5 \
        --thres-file runs/.../thres.txt \
        --routing-policy prev --init-route super \
        --project ./runs/kitti/tracking/anydepth-yolov12l-dynamic
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
from ultralytics.utils.torch_utils import unwrap_model

# Re-use parsers / AP / energy from the baseline video script
from eval_video_kitti import (
    KITTI_TO_BDD100K,
    BDD100K_NAMES,
    EVAL_CLASSES,
    parse_kitti_labels,
    compute_iou,
    compute_ap,
    evaluate_frame,
    compute_map,
    EnergyMonitor,
    draw_boxes,
)


# =============================================================================
# FLOPs pre-computation (one-off per (imgsz, skip-config))
# =============================================================================
def _compute_gflops(model, imgsz_hw, skip_list, device):
    """Compute GFLOPs of the full detector with a fixed skip_list via fvcore."""
    from fvcore.nn import FlopCountAnalysis

    class _Wrap(torch.nn.Module):
        def __init__(self, m, skip):
            super().__init__()
            self.m = m
            self.skip = skip

        def forward(self, x):
            return self.m(x, skip=self.skip)

    wrap = _Wrap(model, skip_list).to(device).eval()
    h, w = imgsz_hw
    dummy = torch.zeros(1, 3, h, w, device=device)
    flops = FlopCountAnalysis(wrap, dummy)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9  # GFLOPs (fvcore MACs; we report MACs as GFLOPs for consistency w/ trainer log)


def _compute_router_gflops(model, imgsz_hw, device):
    """GFLOPs for the shared-tap forward + router (constant per frame)."""
    from fvcore.nn import FlopCountAnalysis

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            feat = self.m._forward_shared_tap(x)
            return self.m.router(feat)

    wrap = _Wrap(model).to(device).eval()
    h, w = imgsz_hw
    dummy = torch.zeros(1, 3, h, w, device=device)
    flops = FlopCountAnalysis(wrap, dummy)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9


# =============================================================================
# letterbox-like resize for router forward (we use ultralytics' predict for
# the detector, so preprocessing there is automatic. The router needs its own
# preprocessed tensor that matches detector's input).
# =============================================================================
def _preprocess_for_router(frame_bgr, imgsz, stride=32, device='cuda', half=False):
    """Minimal letterbox so the router sees the same shape as the detector."""
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(new_shape=imgsz, auto=False, stride=stride)
    img = lb(image=frame_bgr)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC BGR -> CHW RGB
    img = np.ascontiguousarray(img)
    t = torch.from_numpy(img).to(device)
    t = t.half() if half else t.float()
    t = t / 255.0
    return t.unsqueeze(0)


# =============================================================================
# Per-sequence routed evaluation at a single threshold
# =============================================================================
def run_sequence_routed(model_wrap, seq_id, kitti_root, args, energy_monitor,
                        dy_thres, gflops_super, gflops_base, gflops_router,
                        out_root):
    inner = unwrap_model(model_wrap.model)
    device = next(inner.parameters()).device

    img_dir = Path(kitti_root) / 'training' / 'image_02' / seq_id
    label_path = Path(kitti_root) / 'training' / 'label_02' / f'{seq_id}.txt'
    if not img_dir.exists() or not label_path.exists():
        print(f"[!] skip seq {seq_id}: missing data")
        return None

    gt_by_frame = parse_kitti_labels(label_path)
    frame_files = sorted(img_dir.glob('*.png')) or sorted(img_dir.glob('*.jpg'))
    n_frames = len(frame_files)
    print(f"\n[*] Seq {seq_id} @ T={dy_thres:.4f}: {n_frames} frames")

    sample_img = cv2.imread(str(frame_files[0]))
    h, w = sample_img.shape[:2]

    out_dir = Path(out_root) / f'T{dy_thres:.4f}' / f'seq_{seq_id}'
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(str(out_dir / 'output.mp4'), fourcc, 10, (w, h))

    latency_log = []
    energy_log = []
    gflops_log = []
    score_log = []          # score_t computed AT frame t (used to decide skip_{t+1})
    route_log = []          # actual route used at frame t ('S' or 'B')
    all_match_results = []
    all_gt_counts = defaultdict(int)
    warmup_frames = min(10, n_frames)
    num_skip = inner.num_skippable_layers

    # ultralytics stride for letterbox (same as detector predictor)
    stride = int(getattr(inner, 'stride', torch.tensor([32])).max())

    # ── Routing policy ───────────────────────────────────────────────────
    # 'prev' : skip_t is decided by router(frame_{t-1}); frame 0 seeded by --init-route
    # 'same' : skip_t is decided by router(frame_t); router is ON the critical path
    policy = args.routing_policy
    assert policy in ('prev', 'same'), f'invalid --routing-policy: {policy}'

    if policy == 'prev':
        if args.init_route == 'super':
            next_skip = [False] * num_skip
        elif args.init_route == 'base':
            next_skip = [True] * num_skip
        else:
            raise ValueError(f'invalid --init-route: {args.init_route}')

    for idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        frame_idx = int(frame_path.stem)

        power_before = energy_monitor.get_power_mw()

        if policy == 'same':
            # ── (A) SAME-FRAME: router decides, then detector runs ───────
            router_t0 = time.perf_counter()
            router_in = _preprocess_for_router(frame, args.imgsz, stride=stride,
                                               device=device, half=False)
            with torch.no_grad():
                score = inner.predict(router_in, get_score=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            router_ms = (time.perf_counter() - router_t0) * 1000.0
            s_val = float(score.view(-1)[0])
            use_super = s_val >= dy_thres
            skip = [False] * num_skip if use_super else [True] * num_skip
            # score logged at frame t corresponds to route taken at frame t
            score_for_log = s_val

            results = model_wrap.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf,
                verbose=False, skip=skip,
            )
            result = results[0]
            det_ms = result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']
        else:
            # ── (B) PREV-FRAME: detector uses last frame's decision, then router ──
            skip = next_skip
            use_super = not any(skip)
            results = model_wrap.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf,
                verbose=False, skip=skip,
            )
            result = results[0]
            det_ms = result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']

            # Router forward on current frame -> decides NEXT frame
            router_t0 = time.perf_counter()
            router_in = _preprocess_for_router(frame, args.imgsz, stride=stride,
                                               device=device, half=False)
            with torch.no_grad():
                score = inner.predict(router_in, get_score=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            router_ms = (time.perf_counter() - router_t0) * 1000.0
            s_val = float(score.view(-1)[0])
            next_use_super = s_val >= dy_thres
            next_skip = [False] * num_skip if next_use_super else [True] * num_skip
            score_for_log = s_val  # score produced AT frame t (used for t+1)

        power_after = energy_monitor.get_power_mw()

        # ── Latency / Energy / GFLOPs bookkeeping ─────────────────────────
        # same-frame: router is ALWAYS on critical path -> always include
        # prev-frame: router is OFF critical path -> optional include
        if policy == 'same' or args.include_router_in_latency:
            total_ms = det_ms + router_ms
        else:
            total_ms = det_ms

        avg_power_mw = (power_before + power_after) / 2.0
        energy_mj = avg_power_mw * ((det_ms + router_ms) / 1000.0)
        frame_gflops = (gflops_super if use_super else gflops_base) + gflops_router

        is_warmup = idx < warmup_frames
        if not is_warmup:
            latency_log.append(total_ms)
            energy_log.append(energy_mj)
            gflops_log.append(frame_gflops)
            score_log.append(score_for_log)
            route_log.append('S' if use_super else 'B')

        # Extract predictions
        preds = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                preds.append((cls_id, conf, x1, y1, x2, y2))

        gts = gt_by_frame.get(frame_idx, [])
        match_results, gt_count = evaluate_frame(preds, gts, iou_threshold=0.5)
        all_match_results.extend(match_results)
        for cls_id, cnt in gt_count.items():
            all_gt_counts[cls_id] += cnt

        annotated = draw_boxes(frame, preds, gts)
        tag = 'SUPER' if use_super else 'BASE'
        score_label = 'score_t' if policy == 'same' else 'prevScore'
        cv2.putText(annotated,
                    f'{tag}  {score_label}={score_for_log:.3f}  T={dy_thres:.3f}  [{policy}-frame routing]',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        video_out.write(annotated)

        if (idx + 1) % 50 == 0 or idx == n_frames - 1:
            print(f"  [{idx+1}/{n_frames}] ran={tag}  {score_label}={score_for_log:.3f}  "
                  f"det={det_ms:.1f}ms  router={router_ms:.1f}ms  "
                  f"reported_lat={total_ms:.1f}ms  GF={frame_gflops:.2f}")

    video_out.release()

    # Aggregate
    ap_per_class, mean_ap = compute_map(all_match_results, dict(all_gt_counts), [0.5])
    mean_latency = float(np.mean(latency_log)) if latency_log else 0.0
    mean_fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    mean_energy = float(np.mean(energy_log)) if energy_log else 0.0
    mean_gflops = float(np.mean(gflops_log)) if gflops_log else 0.0
    super_ratio = route_log.count('S') / max(len(route_log), 1)

    summary = {
        'sequence': seq_id,
        'threshold': dy_thres,
        'total_frames': n_frames,
        'eval_frames': n_frames - warmup_frames,
        'SUPER_ratio': round(super_ratio, 4),
        'BASE_ratio': round(1 - super_ratio, 4),
        'mean_AP50': round(float(mean_ap), 4),
        'AP50_per_class': {BDD100K_NAMES.get(k, str(k)): round(v, 4) for k, v in ap_per_class.items()},
        'mean_GFLOPs': round(mean_gflops, 3),
        'mean_FPS': round(mean_fps, 2),
        'mean_latency_ms': round(mean_latency, 2),
        'mean_energy_mJ': round(mean_energy, 2),
        'score_stats': {
            'min': round(float(np.min(score_log)), 4) if score_log else 0.0,
            'max': round(float(np.max(score_log)), 4) if score_log else 0.0,
            'mean': round(float(np.mean(score_log)), 4) if score_log else 0.0,
        },
        'gflops_components': {
            'super': round(gflops_super, 3),
            'base': round(gflops_base, 3),
            'router': round(gflops_router, 3),
        },
        'routing_policy': args.routing_policy,
        'init_route': args.init_route if args.routing_policy == 'prev' else 'n/a',
        'latency_includes_router': bool(args.include_router_in_latency or args.routing_policy == 'same'),
    }

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # per-frame CSV
    with open(out_dir / 'per_frame.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rel_idx', 'route', 'score', 'latency_ms', 'gflops', 'energy_mJ'])
        for i, (r, s, lat, gf, en) in enumerate(zip(route_log, score_log, latency_log, gflops_log, energy_log)):
            w.writerow([i + warmup_frames, r, round(s, 4), round(lat, 2), round(gf, 3), round(en, 2)])

    return summary


# =============================================================================
# Top-level driver
# =============================================================================
def _parse_thresholds(args):
    if args.thres:
        return [float(x) for x in args.thres.split(',') if x.strip()]
    if args.thres_file:
        with open(args.thres_file) as f:
            return [float(x) for x in f.read().strip().split()]
    raise ValueError("provide --thres or --thres-file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight', required=True, help='Step 2 checkpoint (with router)')
    ap.add_argument('--kitti_root', default='/media/data/kitti-tracking')
    ap.add_argument('--sequences', nargs='+', default=None)
    ap.add_argument('--imgsz', type=int, nargs='+', default=[192, 640],
                    help='Single value (square) or two (h w). KITTI default: 192 640')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--task', default='detect')
    ap.add_argument('--project', default='./runs/kitti-tracking/dynamic_eval')
    ap.add_argument('--thres', type=str, default='',
                    help='Comma-separated thresholds, e.g. "0.3,0.5,0.7"')
    ap.add_argument('--thres-file', type=str, default='',
                    help='Text file from get_dynamic_thres.py (space-separated)')
    ap.add_argument('--thres-label', type=str, default='',
                    help='Optional label like "bdd" or "kitti" — just goes into folder name')
    ap.add_argument('--routing-policy', choices=['prev', 'same'], default='prev',
                    help='prev: skip_t from router(frame_{t-1}), router off critical path (real-time). '
                         'same: skip_t from router(frame_t), router on critical path (DynamicDet-style).')
    ap.add_argument('--init-route', choices=['super', 'base'], default='super',
                    help='Skip config for frame 0 (prev-frame routing only; ignored for same-frame)')
    ap.add_argument('--include-router-in-latency', action='store_true',
                    help='prev-frame only: include router latency in reported total. '
                         'Default for prev: detector only. For same-frame, router is always included.')
    args = ap.parse_args()

    # imgsz
    if len(args.imgsz) == 1:
        args.imgsz = args.imgsz[0]
        imgsz_hw = (args.imgsz, args.imgsz)
    elif len(args.imgsz) == 2:
        args.imgsz = tuple(args.imgsz)
        imgsz_hw = args.imgsz
    else:
        raise ValueError('--imgsz expects 1 or 2 values')

    thresholds = _parse_thresholds(args)

    # auto-detect sequences
    if args.sequences is None:
        args.sequences = sorted([f.stem for f in (Path(args.kitti_root) / 'training' / 'label_02').glob('*.txt')])
        print(f'[*] auto-detected {len(args.sequences)} sequences')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag_parts = [timestamp, f'policy-{args.routing_policy}']
    if args.thres_label:
        tag_parts.append(args.thres_label)
    run_tag = '_'.join(tag_parts)
    out_root = Path(args.project) / run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f'[*] loading model: {args.weight}')
    model_wrap = YOLO(args.weight, task=args.task)
    inner = unwrap_model(model_wrap.model)
    assert hasattr(inner, 'router'), "checkpoint has no `router` — run Step 2 first"
    num_skip = inner.num_skippable_layers

    device = next(inner.parameters()).device
    if device.type == 'cpu':
        # push to GPU if available — matches eval_video_kitti.py defaults
        if torch.cuda.is_available():
            inner.to('cuda')
            device = torch.device('cuda')

    # Pre-compute GFLOPs (one-off)
    print('[*] computing GFLOPs (SUPER / BASE / router)...')
    try:
        gflops_super = _compute_gflops(inner, imgsz_hw, [False] * num_skip, device)
        gflops_base = _compute_gflops(inner, imgsz_hw, [True] * num_skip, device)
        gflops_router = _compute_router_gflops(inner, imgsz_hw, device)
        print(f'    SUPER : {gflops_super:.3f} GFLOPs')
        print(f'    BASE  : {gflops_base:.3f} GFLOPs')
        print(f'    ROUTER: {gflops_router:.4f} GFLOPs')
    except Exception as e:
        print(f'[!] fvcore FLOPs failed: {e} — reporting 0')
        gflops_super = gflops_base = gflops_router = 0.0

    energy_monitor = EnergyMonitor()

    # Run for each threshold
    all_runs = {}
    for T in thresholds:
        print(f'\n{"="*70}\n[*] threshold = {T}\n{"="*70}')
        per_seq = []
        for seq_id in args.sequences:
            s = run_sequence_routed(
                model_wrap, seq_id, args.kitti_root, args, energy_monitor,
                T, gflops_super, gflops_base, gflops_router, out_root,
            )
            if s:
                per_seq.append(s)
                print(f'  >> seq {seq_id}  SUPER={s["SUPER_ratio"]*100:.1f}%  '
                      f'AP50={s["mean_AP50"]:.4f}  GF={s["mean_GFLOPs"]:.2f}  '
                      f'FPS={s["mean_FPS"]:.1f}  lat={s["mean_latency_ms"]:.1f}ms  '
                      f'E={s["mean_energy_mJ"]:.1f}mJ')

        # Aggregate across sequences for this threshold
        if per_seq:
            overall = {
                'threshold': T,
                'SUPER_ratio': round(float(np.mean([s['SUPER_ratio'] for s in per_seq])), 4),
                'BASE_ratio': round(float(np.mean([s['BASE_ratio'] for s in per_seq])), 4),
                'mean_AP50': round(float(np.mean([s['mean_AP50'] for s in per_seq])), 4),
                'mean_GFLOPs': round(float(np.mean([s['mean_GFLOPs'] for s in per_seq])), 3),
                'mean_FPS': round(float(np.mean([s['mean_FPS'] for s in per_seq])), 2),
                'mean_latency_ms': round(float(np.mean([s['mean_latency_ms'] for s in per_seq])), 2),
                'mean_energy_mJ': round(float(np.mean([s['mean_energy_mJ'] for s in per_seq])), 2),
                'num_sequences': len(per_seq),
                'total_frames': sum(s['total_frames'] for s in per_seq),
                'per_sequence': per_seq,
            }
            all_runs[f'T={T:.4f}'] = overall
            print(f'\n  [T={T:.4f}] OVERALL  SUPER={overall["SUPER_ratio"]*100:.1f}%  '
                  f'AP50={overall["mean_AP50"]:.4f}  GF={overall["mean_GFLOPs"]:.2f}  '
                  f'FPS={overall["mean_FPS"]:.1f}  lat={overall["mean_latency_ms"]:.1f}ms  '
                  f'E={overall["mean_energy_mJ"]:.1f}mJ')

    # Save combined summary
    combined = {
        'timestamp': timestamp,
        'weight': args.weight,
        'imgsz': list(imgsz_hw),
        'conf': args.conf,
        'sequences': args.sequences,
        'gflops_components': {
            'super': round(gflops_super, 3),
            'base': round(gflops_base, 3),
            'router': round(gflops_router, 4),
        },
        'thresholds': thresholds,
        'by_threshold': all_runs,
    }
    with open(out_root / 'overall_summary.json', 'w') as f:
        json.dump(combined, f, indent=2)

    # Print final comparison table
    print(f'\n{"="*96}')
    print(f'[policy={args.routing_policy}]')
    print(f'{"T":>8} {"SUPER%":>8} {"AP50":>8} {"GFLOPs":>10} {"FPS":>8} {"lat(ms)":>10} {"E(mJ)":>10}')
    print('-' * 96)
    for tag, v in all_runs.items():
        T = v['threshold']
        print(f'{T:>8.4f} {v["SUPER_ratio"]*100:>7.1f}% {v["mean_AP50"]:>8.4f} '
              f'{v["mean_GFLOPs"]:>10.2f} {v["mean_FPS"]:>8.1f} '
              f'{v["mean_latency_ms"]:>10.2f} {v["mean_energy_mJ"]:>10.1f}')
    print(f'{"="*96}')
    print(f'[*] results: {out_root}')


if __name__ == '__main__':
    main()


'''
# ── Threshold extraction (run once per source) ─────────────────────────────
# (a) From BDD100K val:
python get_dynamic_thres.py \
    --weight runs/bdd100k/step2/.../step2_router/weights/last.pt \
    --data bdd100k.yaml --imgsz 1280 --split val \
    --out runs/.../thres_bdd.txt

# (b) From KITTI train sequences:
python get_dynamic_thres_kitti.py \
    --weight runs/bdd100k/step2/.../step2_router/weights/last.pt \
    --kitti_root /media/data/kitti-tracking \
    --sequences 0000 0001 0002 0003 --imgsz 192 640 \
    --out runs/.../thres_kitti.txt

# ── KITTI video evaluation — four comparison runs ──────────────────────────
# Policy (B) prev-frame routing × BDD thresholds
python eval_video_kitti_dynamic.py \
    --weight runs/.../step2_router/weights/last.pt \
    --kitti_root /media/data/kitti-tracking --imgsz 192 640 --conf 0.5 \
    --thres-file runs/.../thres_bdd.txt --thres-label thresFromBDD \
    --routing-policy prev --init-route super \
    --project ./runs/kitti/tracking/anydepth-yolov12l-dynamic

# Policy (B) prev-frame routing × KITTI thresholds
python eval_video_kitti_dynamic.py \
    --weight runs/.../step2_router/weights/last.pt \
    --kitti_root /media/data/kitti-tracking --imgsz 192 640 --conf 0.5 \
    --thres-file runs/.../thres_kitti.txt --thres-label thresFromKITTI \
    --routing-policy prev --init-route super \
    --project ./runs/kitti/tracking/anydepth-yolov12l-dynamic

# Policy (A) same-frame routing × BDD thresholds  (DynamicDet-style baseline)
python eval_video_kitti_dynamic.py \
    --weight runs/.../step2_router/weights/last.pt \
    --kitti_root /media/data/kitti-tracking --imgsz 192 640 --conf 0.5 \
    --thres-file runs/.../thres_bdd.txt --thres-label thresFromBDD \
    --routing-policy same \
    --project ./runs/kitti/tracking/anydepth-yolov12l-dynamic

# Policy (A) same-frame routing × KITTI thresholds
python eval_video_kitti_dynamic.py \
    --weight runs/.../step2_router/weights/last.pt \
    --kitti_root /media/data/kitti-tracking --imgsz 192 640 --conf 0.5 \
    --thres-file runs/.../thres_kitti.txt --thres-label thresFromKITTI \
    --routing-policy same \
    --project ./runs/kitti/tracking/anydepth-yolov12l-dynamic
'''
