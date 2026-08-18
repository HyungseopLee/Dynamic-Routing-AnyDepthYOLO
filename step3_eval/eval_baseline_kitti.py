"""
Dynamic-depth video evaluation on KITTI Tracking.

For every frame in every sequence we run BOTH Super-net and Base-net forwards
(latency for each is measured). Then for several "strategies" we record, frame
by frame, which model would have been chosen, and accumulate (pred, GT) into
that strategy's pool.  After all sequences we compute dataset-level mAP@50 for
each strategy and plot the mAP vs latency trade-off.

Strategies evaluated:
  - always_super        : Super every frame
  - always_base         : Base every frame
  - random50            : 50% Base, 50% Super  (seeded RNG)
  - rule_t{TAU}         : if prev-frame's "used model" mean_conf_10 > TAU
                          then Base else Super; first frame = Base
                          for TAU in [0.1, 0.2, ..., 0.9]

Outputs (project dir):
  - per_strategy_summary.json   mAP / mean_latency / %base / etc. per strategy
  - trade_off.png               mAP vs mean latency
  - per_frame_log.csv           per-frame chosen model per strategy (optional)


Usage:
    mkdir -p ./runs/kitti/tracking/baseline
    python -u eval_baseline_video_kitti.py \
        --weight results/step1_finetune/weights/kitti/best.pt \
        --kitti_root /media/data/kitti-tracking \
        --imgsz 384 1248 \
        --conf 0.001 \
        --project ./runs/kitti/tracking/baseline \
        2>&1 | tee ./runs/kitti/tracking/baseline/eval.log


"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.ERROR)


# ---- KITTI Tracking label string -> our KITTI-finetuned model class index ----
# Same indexing as ultralytics/cfg/datasets/kitti.yaml (7 classes, no Misc/DontCare).
KITTI_TO_BDD = {                   # name kept for backward compat in code below
    "Car": 0, "Van": 1, "Truck": 2,
    "Pedestrian": 3, "Person_sitting": 4,
    "Cyclist": 5, "Tram": 6,
}
BDD_NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian",
             4: "person_sitting", 5: "cyclist", 6: "tram"}
EVAL_CLS = sorted(set(KITTI_TO_BDD.values()))   # [0, 1, 2, 3, 4, 5, 6]

# Rule-based parameters
# Confidence features used by rule strategies. Mirrors the offline analysis CSV
# (tools/per_image_loss_pr_conf.py): same thresholds, same top-K's, so plot
# tooling can pick any feature without having to re-run the eval.
#   - top{K}_mean_conf  : mean of top-K highest post-NMS confidences
#   - mean_conf_ge_{T}  : mean of post-NMS confidences >= T (T = 0.01 ... 0.75)
#   - mean_conf_all     : mean of all post-NMS confidences (no threshold)
CONF_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75)
CONF_TOPKS = (10, 20)                                       # top30 dropped (speed)
RULE_CONF_FEATURES = (
    [f"top{k}_mean_conf" for k in CONF_TOPKS]
    + [f"mean_conf_ge_{int(round(t*100)):02d}" for t in CONF_THRESHOLDS]
    + ["mean_conf_all"]
)
RULE_TAU_GRID = [round(0.10 * i, 2) for i in range(1, 10)]  # 0.1..0.9 step 0.10 (was 0.05)

# Learned router: tap layer indices in the AnyDepth-YOLO model that the
# router was trained on (must match tools/dump_router_features.py).
ROUTER_HOOK_LAYERS = (4, 6, 8, 14, 17, 20)
# τ grid for the learned router's Δ̂ output (regression on cls_diff scale).
ROUTER_TAU_GRID = [-0.20, -0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20]


import torch.nn as _nn
class RouterMLP(_nn.Module):
    """Mirror of tools/train_router.py:RouterMLP for inference."""
    def __init__(self, in_dim, hidden=(256, 64), dropout=0.2):
        super().__init__()
        layers = [_nn.BatchNorm1d(in_dim)]
        prev = in_dim
        for h in hidden:
            layers += [_nn.Linear(prev, h), _nn.BatchNorm1d(h),
                       _nn.ReLU(inplace=True), _nn.Dropout(dropout)]
            prev = h
        layers += [_nn.Linear(prev, 1)]
        self.net = _nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)
RANDOM_SEED = 0


# ---- helpers ----
def parse_kitti_labels(label_path):
    """Return (gt_by_frame, dontcare_by_frame).
    gt_by_frame[fi]   = [(cls_bdd, x1, y1, x2, y2), ...]   evaluable GT
    dontcare_by_frame[fi] = [(x1, y1, x2, y2), ...]       ignore regions
    """
    gt = defaultdict(list)
    dont_care = defaultdict(list)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10: continue
            frame_idx = int(parts[0])
            obj_type = parts[2]
            x1, y1, x2, y2 = map(float, parts[6:10])
            if obj_type == "DontCare":
                dont_care[frame_idx].append((x1, y1, x2, y2))
                continue
            if obj_type not in KITTI_TO_BDD: continue
            cls = KITTI_TO_BDD[obj_type]
            gt[frame_idx].append((cls, x1, y1, x2, y2))
    return gt, dont_care


def filter_dontcare(preds, dc_boxes, ioa_th=0.5):
    """Drop predictions whose box has Intersection-over-PredArea (IoA) >= ioa_th
    with any DontCare region. This is KITTI's standard ignore rule.
    """
    if not dc_boxes:
        return preds
    keep = []
    for p in preds:
        px1, py1, px2, py2 = p[2:6]
        p_area = max(0.0, (px2 - px1)) * max(0.0, (py2 - py1))
        ignore = False
        if p_area > 0:
            for dx1, dy1, dx2, dy2 in dc_boxes:
                ix1, iy1 = max(px1, dx1), max(py1, dy1)
                ix2, iy2 = min(px2, dx2), min(py2, dy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter / p_area >= ioa_th:
                    ignore = True; break
        if not ignore:
            keep.append(p)
    return keep


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


IOU_GRID = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # 0.5..0.95 step 0.05


def match_frame_multi_iou(preds, gts, iou_grid=IOU_GRID):
    """For each pred, return (cls, conf, [is_tp@iou for iou in iou_grid]).
    Greedy matching by descending conf, each GT used once per IoU threshold.
    Also returns gt count per class.
    """
    gt_count = defaultdict(int)
    for gc, *_ in gts:
        if gc in EVAL_CLS: gt_count[gc] += 1
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    # pre-compute IoU between every pred and every gt, only for class-matched pairs
    matched = [[False] * len(gts) for _ in iou_grid]
    results = []
    for pc, conf, px1, py1, px2, py2 in preds_sorted:
        if pc not in EVAL_CLS:
            continue
        # find best gt per IoU threshold (greedy with class match)
        # compute pairwise IoUs once
        ious = []
        for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
            if gc != pc:
                ious.append(-1.0)
            else:
                ious.append(iou_xyxy((px1, py1, px2, py2), (gx1, gy1, gx2, gy2)))
        tp_flags = []
        for ti, iou_th in enumerate(iou_grid):
            best_iou, best_idx = 0.0, -1
            for gi, v in enumerate(ious):
                if v < 0 or matched[ti][gi]:
                    continue
                if v > best_iou:
                    best_iou = v; best_idx = gi
            is_tp = best_iou >= iou_th and best_idx >= 0
            if is_tp:
                matched[ti][best_idx] = True
            tp_flags.append(is_tp)
        results.append((pc, conf, tp_flags))
    return results, dict(gt_count)


def match_frame(preds, gts, iou_th=0.5):
    """Legacy single-IoU helper (kept for backward compat). Returns (cls, conf, is_tp)."""
    res_multi, gt_count = match_frame_multi_iou(preds, gts, iou_grid=[iou_th])
    return [(c, conf, flags[0]) for (c, conf, flags) in res_multi], gt_count


def compute_ap(precisions, recalls):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = mpre[mrec >= t]
        ap += (p.max() if len(p) > 0 else 0.0) / 101
    return ap


def dataset_map_multi_iou(all_matches_multi, all_gt_counts, iou_grid=IOU_GRID):
    """all_matches_multi: list of (cls, conf, [is_tp@iou for iou in iou_grid])."""
    ap_iou_cls = {}                          # iou_idx -> {cls: ap}
    for ti, iou_th in enumerate(iou_grid):
        ap_iou_cls[ti] = {}
        for cls in EVAL_CLS:
            items = [(c, flags[ti]) for (cl, c, flags) in all_matches_multi if cl == cls]
            n_gt = all_gt_counts.get(cls, 0)
            if n_gt == 0: continue
            items.sort(key=lambda x: x[0], reverse=True)
            tp_c, fp_c = 0, 0; precs, recs = [], []
            for conf, is_tp in items:
                if is_tp: tp_c += 1
                else:     fp_c += 1
                precs.append(tp_c / (tp_c + fp_c))
                recs.append(tp_c / n_gt)
            ap_iou_cls[ti][cls] = compute_ap(np.array(precs), np.array(recs))
    # AP@50 per class = ap_iou_cls[0]
    ap50_per_cls = ap_iou_cls[0]
    map50 = float(np.mean(list(ap50_per_cls.values()))) if ap50_per_cls else 0.0
    # mAP@[0.5:0.95] = mean over IoUs of (mean over classes of AP)
    per_iou_map = []
    for ti in range(len(iou_grid)):
        d = ap_iou_cls[ti]
        if d: per_iou_map.append(float(np.mean(list(d.values()))))
        else: per_iou_map.append(0.0)
    map5095 = float(np.mean(per_iou_map))
    return ap50_per_cls, map50, map5095


def dataset_map50(all_matches, all_gt_counts):
    """Legacy single-IoU mAP (kept for backward compat)."""
    aps = {}
    for cls in EVAL_CLS:
        items = [(c, tp) for (cl, c, tp) in all_matches if cl == cls]
        n_gt = all_gt_counts.get(cls, 0)
        if n_gt == 0: continue
        items.sort(key=lambda x: x[0], reverse=True)
        tp_c, fp_c = 0, 0; precs, recs = [], []
        for conf, is_tp in items:
            if is_tp: tp_c += 1
            else:     fp_c += 1
            precs.append(tp_c / (tp_c + fp_c))
            recs.append(tp_c / n_gt)
        aps[cls] = compute_ap(np.array(precs), np.array(recs))
    return aps, (float(np.mean(list(aps.values()))) if aps else 0.0)


def conf_mean_ge(conf_tensor, t):
    if conf_tensor.numel() == 0: return 0.0
    m = conf_tensor >= t
    return float(conf_tensor[m].mean().item()) if int(m.sum()) > 0 else 0.0


def conf_top_k_mean(conf_tensor, k):
    if conf_tensor.numel() == 0: return 0.0
    k_eff = min(int(k), conf_tensor.numel())
    top = torch.topk(conf_tensor, k_eff).values
    return float(top.mean().item())


def compute_conf_features(conf_tensor):
    """Return dict[feature_name] -> float for one frame's predicted confidences.

    Keys mirror RULE_CONF_FEATURES (CONF_THRESHOLDS + CONF_TOPKS + mean_conf_all).
    """
    feats = {}
    for k in CONF_TOPKS:
        feats[f"top{k}_mean_conf"] = conf_top_k_mean(conf_tensor, k)
    for t in CONF_THRESHOLDS:
        feats[f"mean_conf_ge_{int(round(t*100)):02d}"] = conf_mean_ge(conf_tensor, t)
    feats["mean_conf_all"] = (float(conf_tensor.mean().item())
                              if conf_tensor.numel() > 0 else 0.0)
    return feats


def measure_flops_super_base(model, imgsz, device):
    """Profile GFLOPs for Super (no skip) and Base (skip all) forwards via thop.

    Wraps the model in a small forward(x) module with a fixed skip pattern so
    thop.profile sees a normal `forward(x)` signature.
    Returns (gflops_super, gflops_base). (0, 0) on failure.
    """
    try:
        import thop
        from copy import deepcopy
        import torch.nn as nn
        num_skip = getattr(model, "num_skippable_layers", 0)
        H, W = int(imgsz[0]), int(imgsz[1])     # imgsz is now (H, W) -- matches eval_policy_video_kitti.py
        H = ((H + 31) // 32) * 32
        W = ((W + 31) // 32) * 32
        x = torch.empty((1, 3, H, W), device=device)

        class _Wrap(nn.Module):
            def __init__(self, m, skip_pattern):
                super().__init__()
                self.m = m
                self.skip_pattern = skip_pattern
            def forward(self, x):
                return self.m._predict_once(x, False, False, None,
                                            skip=self.skip_pattern,
                                            return_features=False)

        def _gflops(skip_pattern):
            wrapped = _Wrap(deepcopy(model).eval(), skip_pattern).to(device)
            macs, _ = thop.profile(wrapped, inputs=(x,), verbose=False)
            return float(macs) * 2 / 1e9

        gflops_s = _gflops([False] * num_skip)
        gflops_b = _gflops([True]  * num_skip)
        return gflops_s, gflops_b
    except Exception as e:
        print(f"[!] FLOPs measurement failed: {e}")
        return 0.0, 0.0


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
            print("[!] pynvml not available; energy = 0 mJ")

    def power_mw(self):
        if not self.available: return 0.0
        return float(self.pynvml.nvmlDeviceGetPowerUsage(self.handle))


def boxes_to_preds(r):
    """Ultralytics Result → list of (cls, conf, x1, y1, x2, y2). xyxy in original-image coords."""
    out = []
    if r.boxes is None or len(r.boxes) == 0: return out
    cls = r.boxes.cls.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    xyxy = r.boxes.xyxy.cpu().numpy()
    for i in range(len(cls)):
        out.append((int(cls[i]), float(conf[i]), float(xyxy[i, 0]), float(xyxy[i, 1]),
                    float(xyxy[i, 2]), float(xyxy[i, 3])))
    return out


# ---- strategies ----
def build_strategies(include_learned=False):
    """Return list of (name, feature, decide_fn).

    feature is the conf-feature key the rule reads from prev_feats (or None for
    non-rule strategies), or the literal string "learned" for the learned router
    (its decide_fn reads prev_value = current-frame Δ̂ from the router output).
    decide_fn(prev_choice, prev_value, frame_idx, rng).
    """
    strats = []
    # random sweep: P(super) = p/100, in steps of 10. p=0 -> always_base, p=100 -> always_super.
    for p in range(0, 101, 10):
        p_super = p / 100.0
        def make_rand(p_super=p_super):
            def rand(prev_choice, prev_value, fi, rng):
                return "super" if rng.random() < p_super else "base"
            return rand
        strats.append((f"random_p{p:03d}", None, make_rand()))
    for feat in RULE_CONF_FEATURES:
        for tau in RULE_TAU_GRID:
            def make_rule(tau=tau):
                def rule(prev_choice, prev_value, fi, rng):
                    if fi == 0: return "base"
                    return "base" if prev_value > tau else "super"
                return rule
            strats.append((f"rule_{feat}_t{int(round(tau*100)):02d}", feat, make_rule(tau)))
    if include_learned:
        for tau in ROUTER_TAU_GRID:
            def make_learned(tau=tau):
                def learned(prev_choice, prev_value, fi, rng):
                    return "super" if prev_value > tau else "base"
                return learned
            strats.append((f"learned_t{int(round(tau*100)):+03d}", "learned", make_learned(tau)))
    return strats


# ---- per-strategy state ----
class StrategyState:
    def __init__(self, name):
        self.name = name
        self.matches_multi = []  # list of (cls, conf, [is_tp@iou for iou in IOU_GRID])
        self.gt_count = defaultdict(int)
        self.latency_ms = []     # per-frame latency
        self.energy_mj  = []     # per-frame energy
        self.n_base = 0
        self.n_super = 0
        self.prev_choice = None
        self.prev_feats = {}


# ---- main per-sequence loop ----
def run_sequence(yolo, seq, kitti_root, args, strategies, rng, energy_monitor,
                 router=None, router_captured=None):
    """Return dict[strategy_name] -> StrategyState filled for THIS sequence only."""
    img_dir = Path(kitti_root) / "training" / "image_02" / seq
    label_path = Path(kitti_root) / "training" / "label_02" / f"{seq}.txt"
    seq_states = {name: StrategyState(name) for name, _f, _d in strategies}
    if not img_dir.exists() or not label_path.exists():
        print(f"[!] skip {seq}: missing"); return seq_states
    gt_by_frame, dontcare_by_frame = parse_kitti_labels(label_path)
    frames = sorted(img_dir.glob("*.png")) or sorted(img_dir.glob("*.jpg"))
    if args.limit > 0:
        frames = frames[:args.limit]
    n = len(frames)
    print(f"\n[seq {seq}] {n} frames")

    num_skip = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super = [False] * num_skip
    skip_base  = [True]  * num_skip

    for fi, fpath in enumerate(frames):
        frame_idx = int(fpath.stem)
        bgr = cv2.imread(str(fpath))
        if bgr is None: continue

        # Super forward (with power sampling)
        p0 = energy_monitor.power_mw()
        t0 = time.perf_counter()
        r_super = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                               iou=0.7, skip=skip_super, verbose=False, device=args.device)[0]
        t1 = time.perf_counter()
        p1 = energy_monitor.power_mw()
        # If a learned router is configured, grab GAP feature from super's
        # hooked layers BEFORE the base forward overwrites them.
        # Measure router overhead (GAP + concat) for the super path.
        router_feat_super = None
        router_oh_ms_super = 0.0
        if router is not None and router_captured:
            t_rs0 = time.perf_counter()
            router_feat_super = torch.cat([
                router_captured[idx].mean(dim=(2, 3)).squeeze(0).float()
                for idx in ROUTER_HOOK_LAYERS
            ], dim=0)
            t_rs1 = time.perf_counter()
            router_oh_ms_super = (t_rs1 - t_rs0) * 1000.0
            router_captured.clear()
        # Base forward
        r_base = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                              iou=0.7, skip=skip_base, verbose=False, device=args.device)[0]
        t2 = time.perf_counter()
        p2 = energy_monitor.power_mw()
        router_feat_base = None
        router_oh_ms_base = 0.0
        if router is not None and router_captured:
            t_rb0 = time.perf_counter()
            router_feat_base = torch.cat([
                router_captured[idx].mean(dim=(2, 3)).squeeze(0).float()
                for idx in ROUTER_HOOK_LAYERS
            ], dim=0)
            t_rb1 = time.perf_counter()
            router_oh_ms_base = (t_rb1 - t_rb0) * 1000.0
            router_captured.clear()
        lat_super_ms = (t1 - t0) * 1000.0
        lat_base_ms  = (t2 - t1) * 1000.0
        # energy = P_avg * t  (mW * s = mJ)
        e_super_mj = 0.5 * (p0 + p1) * (lat_super_ms / 1000.0)
        e_base_mj  = 0.5 * (p1 + p2) * (lat_base_ms  / 1000.0)

        preds_super = boxes_to_preds(r_super)
        preds_base  = boxes_to_preds(r_base)
        conf_t_super = (r_super.boxes.conf.cpu() if (r_super.boxes is not None and len(r_super.boxes))
                        else torch.empty(0))
        conf_t_base  = (r_base.boxes.conf.cpu() if (r_base.boxes is not None and len(r_base.boxes))
                        else torch.empty(0))
        # Measure routing overhead = wall-clock time of compute_conf_features().
        # This is what a deployed dynamic-routing system would pay per frame on
        # top of the chosen model's forward latency.
        t_f0 = time.perf_counter()
        feats_super = compute_conf_features(conf_t_super)
        t_f1 = time.perf_counter()
        feats_base  = compute_conf_features(conf_t_base)
        t_f2 = time.perf_counter()
        routing_overhead_ms_super = (t_f1 - t_f0) * 1000.0
        routing_overhead_ms_base  = (t_f2 - t_f1) * 1000.0

        # Learned router predictions for this frame.
        # delta_super = router applied to super-path features (used if super chosen this frame)
        # delta_base  = router applied to base-path features  (used if base chosen this frame)
        # We profile the MLP forward separately so a deployed learned strategy can
        # be charged (GAP + MLP) on top of the chosen network's forward latency.
        delta_super = delta_base = 0.0
        router_mlp_ms = 0.0
        if router is not None and router_feat_super is not None and router_feat_base is not None:
            with torch.no_grad():
                t_m0 = time.perf_counter()
                stacked = torch.stack([router_feat_super, router_feat_base], dim=0)
                d = router(stacked)
                if d.is_cuda: torch.cuda.synchronize()
                t_m1 = time.perf_counter()
                delta_super = float(d[0].item()); delta_base = float(d[1].item())
            # Per-sample MLP cost (we ran 2 samples; deployment runs 1).
            router_mlp_ms = (t_m1 - t_m0) * 1000.0 / 2.0
        # Total learned-router overhead per frame = GAP(chosen_path) + MLP(1 sample).
        learned_oh_ms_super = router_oh_ms_super + router_mlp_ms
        learned_oh_ms_base  = router_oh_ms_base  + router_mlp_ms

        gts = gt_by_frame.get(frame_idx, [])
        dc_boxes = dontcare_by_frame.get(frame_idx, [])
        # KITTI standard: drop predictions overlapping DontCare regions before scoring
        preds_super = filter_dontcare(preds_super, dc_boxes)
        preds_base  = filter_dontcare(preds_base,  dc_boxes)

        for name, feat, decide in strategies:
            st = seq_states[name]
            prev_value = st.prev_feats.get(feat, 0.0) if feat else 0.0
            choice = decide(st.prev_choice, prev_value, fi, rng)
            if choice == "super":
                preds = preds_super; lat = lat_super_ms; eng = e_super_mj
                this_feats = dict(feats_super)
                this_feats["learned"] = delta_super
                if feat == "learned":
                    routing_oh = learned_oh_ms_super
                elif feat:
                    routing_oh = routing_overhead_ms_super
                else:
                    routing_oh = 0.0
                st.n_super += 1
            else:
                preds = preds_base;  lat = lat_base_ms;  eng = e_base_mj
                this_feats = dict(feats_base)
                this_feats["learned"] = delta_base
                if feat == "learned":
                    routing_oh = learned_oh_ms_base
                elif feat:
                    routing_oh = routing_overhead_ms_base
                else:
                    routing_oh = 0.0
                st.n_base += 1
            # Rule strategies pay conf-feature overhead; learned strategies pay
            # GAP + MLP overhead; random/baseline strategies pay nothing.
            st.latency_ms.append(lat + routing_oh)
            st.energy_mj.append(eng)
            m, gc = match_frame_multi_iou(preds, gts)
            st.matches_multi.extend(m)
            for k, v in gc.items(): st.gt_count[k] += v
            st.prev_choice = choice
            st.prev_feats  = this_feats

        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{n}]  S={lat_super_ms:.1f}ms B={lat_base_ms:.1f}ms")
    return seq_states


def state_to_summary(st, gflops_super=0.0, gflops_base=0.0):
    ap50_per_cls, m_ap50, m_ap5095 = dataset_map_multi_iou(st.matches_multi, dict(st.gt_count))
    n_total = st.n_super + st.n_base
    pct_base = 100.0 * st.n_base / max(n_total, 1)
    mean_lat = float(np.mean(st.latency_ms)) if st.latency_ms else 0.0
    mean_fps = (1000.0 / mean_lat) if mean_lat > 0 else 0.0
    mean_energy = float(np.mean(st.energy_mj)) if st.energy_mj else 0.0
    # weighted FLOPs by base/super mix
    mean_flops = (pct_base / 100.0) * gflops_base + (1 - pct_base / 100.0) * gflops_super
    return {
        "mAP@50":       round(float(m_ap50),   4),
        "mAP@50:95":    round(float(m_ap5095), 4),
        "AP50_per_class": {BDD_NAMES.get(k, str(k)): round(v, 4) for k, v in ap50_per_cls.items()},
        "n_frames":    n_total,
        "n_super":     st.n_super,
        "n_base":      st.n_base,
        "pct_base":    round(pct_base, 2),
        "mean_latency_ms": round(mean_lat,   3),
        "mean_fps":        round(mean_fps,   2),
        "mean_energy_mJ":  round(mean_energy, 3),
        "mean_GFLOPs":     round(mean_flops, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--kitti_root", default="/media/data/kitti-tracking")
    ap.add_argument("--sequences", type=str, nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1248],
                    help="(H, W). Default = KITTI native (1242x375) padded to 32-multiples. "
                         "Same convention as eval_policy_video_kitti.py.")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="NMS conf threshold. Default 0.001 keeps many low-conf "
                         "detections for accurate AP curves. Same as eval_policy_video_kitti.py.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--project", default="./runs/kitti/tracking/dynamic")
    ap.add_argument("--limit", type=int, default=0, help="cap frames per seq (debug)")
    ap.add_argument("--router_ckpt", default="", help="path to learned router .pt; if set, add learned_tXX strategies")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = Path(args.project) / ts
    project.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading {args.weight}")
    yolo = YOLO(args.weight, task="detect")
    yolo.model.to(args.device).eval()
    num_skip = getattr(yolo.model, "num_skippable_layers", 0)
    if num_skip <= 0:
        raise SystemExit("AnyDepth model required.")
    print(f"[*] num_skippable_layers={num_skip}, imgsz={args.imgsz}, conf={args.conf}")

    label_dir = Path(args.kitti_root) / "training" / "label_02"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.txt"))
    print(f"[*] {len(seqs)} sequences")

    # Optional learned router
    router = None
    router_hooks = []
    router_captured = {}
    if args.router_ckpt:
        print(f"[*] Loading router: {args.router_ckpt}")
        ckpt = torch.load(args.router_ckpt, map_location=args.device, weights_only=True)
        router = RouterMLP(in_dim=int(ckpt["in_dim"]),
                           hidden=tuple(ckpt["hidden"]),
                           dropout=float(ckpt["dropout"])).to(args.device).eval()
        router.load_state_dict(ckpt["state_dict"])
        # Install forward hooks on the AnyDepth model to capture features
        # during the Base forward pass (matching dump_router_features.py).
        def _make_hook(idx):
            def hook(_m, _i, out):
                router_captured[idx] = out
            return hook
        for idx in ROUTER_HOOK_LAYERS:
            router_hooks.append(yolo.model.model[idx].register_forward_hook(_make_hook(idx)))
        print(f"[*] Router loaded: in_dim={ckpt['in_dim']}  hidden={ckpt['hidden']}")

    strategies = build_strategies(include_learned=router is not None)
    strat_names = [s[0] for s in strategies]
    print(f"[*] {len(strategies)} strategies")
    rng = __import__("random").Random(RANDOM_SEED)

    # GFLOPs profiling
    gflops_super, gflops_base = measure_flops_super_base(yolo.model, args.imgsz, args.device)
    print(f"[*] GFLOPs  super={gflops_super:.2f}  base={gflops_base:.2f}")

    energy_monitor = EnergyMonitor()

    # GPU warmup (avoid first-frame outlier)
    for _ in range(3):
        yolo.predict(source=np.zeros((args.imgsz[0], args.imgsz[1], 3), dtype=np.uint8),
                     imgsz=tuple(args.imgsz), verbose=False, device=args.device,
                     skip=[False]*num_skip)

    # per-sequence + accumulated states
    per_seq_states = {}                                    # seq -> {name: StrategyState}
    accum_states = {name: StrategyState(name) for name in strat_names}

    for seq in seqs:
        seq_states = run_sequence(yolo, seq, args.kitti_root, args, strategies, rng,
                                  energy_monitor, router=router, router_captured=router_captured)
        per_seq_states[seq] = seq_states
        for name in strat_names:
            ss = seq_states[name]; acc = accum_states[name]
            acc.matches_multi.extend(ss.matches_multi)
            for k, v in ss.gt_count.items(): acc.gt_count[k] += v
            acc.latency_ms.extend(ss.latency_ms)
            acc.energy_mj.extend(ss.energy_mj)
            acc.n_super += ss.n_super
            acc.n_base  += ss.n_base

    # ---- summarize ----
    summary = {"timestamp": ts, "weight": args.weight, "imgsz": args.imgsz,
               "conf": args.conf, "kitti_root": args.kitti_root, "sequences": seqs,
               "rule_features": RULE_CONF_FEATURES, "rule_tau_grid": RULE_TAU_GRID,
               "gflops_super": round(gflops_super, 3), "gflops_base": round(gflops_base, 3),
               "overall": {name: state_to_summary(accum_states[name], gflops_super, gflops_base)
                           for name in strat_names},
               "per_sequence": {seq: {name: state_to_summary(per_seq_states[seq][name],
                                                              gflops_super, gflops_base)
                                      for name in strat_names}
                                for seq in seqs}}
    with open(project / "per_strategy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[*] summary -> {project / 'per_strategy_summary.json'}")
    print("\n=== Overall (all sequences accumulated) ===")
    hdr = f"{'strategy':18s} {'mAP@50':>7s} {'mAP@.5:.95':>10s} {'lat(ms)':>8s} {'FPS':>6s} {'GFLOPs':>7s} {'mJ':>7s} {'%base':>6s}"
    print(hdr)
    for name in strat_names:
        s = summary["overall"][name]
        print(f"  {name:16s} {s['mAP@50']:7.4f} {s['mAP@50:95']:10.4f} "
              f"{s['mean_latency_ms']:8.2f} {s['mean_fps']:6.1f} {s['mean_GFLOPs']:7.2f} "
              f"{s['mean_energy_mJ']:7.2f} {s['pct_base']:5.1f}")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("turbo")
    feat_color = {feat: cmap(i / max(len(RULE_CONF_FEATURES) - 1, 1))
                  for i, feat in enumerate(RULE_CONF_FEATURES)}

    def _parse_rule_name(name):
        """rule_{feature}_t{TT} -> (feature, tau_int) or None."""
        if not name.startswith("rule_"): return None
        body = name[len("rule_"):]
        if "_t" not in body: return None
        feat, tt = body.rsplit("_t", 1)
        try: return feat, int(tt)
        except ValueError: return None

    def _x_value(s, xkey):
        if xkey == "pct_super": return 100.0 - s["pct_base"]
        if xkey == "latency":   return s["mean_latency_ms"]
        raise ValueError(xkey)

    def trade_off_panel(ax, strat_summary, title, xkey="latency", show_legend=False):
        feat_pts = {f: [] for f in RULE_CONF_FEATURES}  # feat -> [(x, y, tau)]
        rand_pts = []
        learned_pts = []  # (x, y, tt)
        ap_super = ap_base = None       # endpoint mAPs for reference lines
        for name, s in strat_summary.items():
            x = _x_value(s, xkey); y = s["mAP@50"]
            parsed = _parse_rule_name(name)
            if parsed is not None:
                feat, tt = parsed
                if feat in feat_pts: feat_pts[feat].append((x, y, tt))
            elif name.startswith("random_p"):
                p = int(name[len("random_p"):])
                rand_pts.append((x, y, p))
                if p == 100: ap_super = y
                elif p == 0: ap_base  = y
            elif name.startswith("learned_t"):
                try:
                    tt = int(name[len("learned_t"):])
                except ValueError:
                    tt = 0
                learned_pts.append((x, y, tt))
        # Reference lines: always-super / always-base mAP@50.
        # Drawn first so strategy curves stay on top.
        if ap_super is not None:
            ax.axhline(ap_super, color="tab:blue", ls=":", lw=1.0, alpha=0.8, zorder=2,
                       label=f"always_super (mAP@50={ap_super:.3f})" if show_legend else None)
        if ap_base is not None:
            ax.axhline(ap_base, color="tab:red", ls=":", lw=1.0, alpha=0.8, zorder=2,
                       label=f"always_base  (mAP@50={ap_base:.3f})" if show_legend else None)
        if rand_pts:
            rand_pts.sort(key=lambda t: t[0])
            rx = [p[0] for p in rand_pts]; ry = [p[1] for p in rand_pts]
            ax.plot(rx, ry, "--s", color="black", lw=1.2, ms=4, zorder=6,
                    label="random sweep (baseline)" if show_legend else None)
        for feat, pts in feat_pts.items():
            if not pts: continue
            pts.sort(key=lambda t: t[0])
            rl = [p[0] for p in pts]; rm = [p[1] for p in pts]
            tt_list = [p[2] for p in pts]
            ax.plot(rl, rm, "-o", color=feat_color[feat], lw=1.0, ms=3, zorder=4,
                    label=feat if show_legend else None, alpha=0.85)
            if xkey == "pct_super":
                for xv, yv, tt in zip(rl, rm, tt_list):
                    ax.annotate(f"τ={tt/100:.2f}", (xv, yv), fontsize=6,
                                xytext=(3, 4), textcoords="offset points",
                                color=feat_color[feat], zorder=7)
        if learned_pts:
            learned_pts.sort(key=lambda t: t[0])
            lx = [p[0] for p in learned_pts]; ly = [p[1] for p in learned_pts]
            ax.plot(lx, ly, "-D", color="crimson", lw=2.0, ms=6, zorder=8,
                    label="learned router (τ sweep)" if show_legend else None)
            if xkey == "pct_super":
                for xv, yv, tt in zip(lx, ly, [p[2] for p in learned_pts]):
                    ax.annotate(f"τ={tt/100:+.2f}", (xv, yv), fontsize=6,
                                xytext=(3, -10), textcoords="offset points",
                                color="crimson", zorder=9)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)

    # Figure 1: per-sequence grid
    n_seq = len(seqs)
    n_cols = 5
    n_rows = int(np.ceil(n_seq / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    for i, seq in enumerate(seqs):
        r, c = divmod(i, n_cols)
        trade_off_panel(axes[r, c], summary["per_sequence"][seq],
                        title=f"seq {seq} ({per_seq_states[seq][strat_names[0]].n_super + per_seq_states[seq][strat_names[0]].n_base} frames)",
                        xkey="latency")
        axes[r, c].set_xlabel("latency (ms)", fontsize=7)
        axes[r, c].set_ylabel("mAP@50", fontsize=7)
    for j in range(n_seq, n_rows * n_cols):
        r, c = divmod(j, n_cols); axes[r, c].axis("off")
    fig.suptitle("KITTI Tracking — per-sequence trade-off  "
                 "(one line per conf-feature τ-sweep; dashed black = random baseline)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(project / "trade_off_per_sequence.png", dpi=140)
    plt.close(fig)
    print(f"[*] figure -> {project / 'trade_off_per_sequence.png'}")

    # Figure 2: overall accumulated — right panel only (%super)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))
    trade_off_panel(ax, summary["overall"],
                    title="accuracy vs %super", xkey="pct_super", show_legend=True)
    ax.set_xlabel("% super-net used  (0 = always base, 100 = always super)")
    ax.set_ylabel("dataset mAP@50  (accumulated)")
    ax.set_xlim(-5, 105)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    fig.suptitle("KITTI Tracking — trade-off (overall accumulated, %super only)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(project / "trade_off_overall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] figure -> {project / 'trade_off_overall.png'}")

    # ===== Figure 3: focused comparison (all rule conf features + baselines) =====
    # Every feature in RULE_CONF_FEATURES is plotted as its own τ-sweep curve so we
    # don't need to re-run eval to compare different conf features.
    _FOCUS_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    _focus_cmap = plt.get_cmap("tab10")
    FOCUS_FEATS = [(f, _focus_cmap(i % 10), _FOCUS_MARKERS[i % len(_FOCUS_MARKERS)])
                   for i, f in enumerate(RULE_CONF_FEATURES)]
    overall = summary["overall"]
    ap_super = overall.get("random_p100", {}).get("mAP@50")
    ap_base  = overall.get("random_p000", {}).get("mAP@50")
    lat_super = overall.get("random_p100", {}).get("mean_latency_ms")
    lat_base  = overall.get("random_p000", {}).get("mean_latency_ms")

    fig, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(15, 6))
    for xkey, ax, xlabel in [("latency", ax_l, "mean latency (ms)"),
                              ("pct_super", ax_p, "% super-net used")]:
        if ap_super is not None:
            ax.axhline(ap_super, color="tab:blue", ls=":", lw=1.2, alpha=0.9,
                       label=f"always_super (mAP@50={ap_super:.3f})")
        if ap_base is not None:
            ax.axhline(ap_base, color="tab:red", ls=":", lw=1.2, alpha=0.9,
                       label=f"always_base (mAP@50={ap_base:.3f})")
        if xkey == "latency":
            if lat_super is not None:
                ax.axvline(lat_super, color="tab:blue", ls=":", lw=0.8, alpha=0.5)
            if lat_base is not None:
                ax.axvline(lat_base,  color="tab:red",  ls=":", lw=0.8, alpha=0.5)
        # Linear-interpolation baseline between always-base and always-super
        # (= expected random_pXX curve, since mAP/lat both average linearly).
        if ap_super is not None and ap_base is not None:
            xs_lin = ys_lin = None
            if xkey == "pct_super":
                xs_lin, ys_lin = [0.0, 100.0], [ap_base, ap_super]
            elif lat_super is not None and lat_base is not None:
                xs_lin, ys_lin = [lat_base, lat_super], [ap_base, ap_super]
            if xs_lin is not None:
                ax.plot(xs_lin, ys_lin, color="gray", ls="-", lw=1.0, alpha=0.6,
                        zorder=2, label="base–super linear interp")
        # Random sweep baseline (random_p000 .. random_p100).
        rand_pts = []
        for name, s in overall.items():
            if not name.startswith("random_p"): continue
            try:
                p = int(name[len("random_p"):])
            except ValueError:
                continue
            rand_pts.append((_x_value(s, xkey), s["mAP@50"], p))
        if rand_pts:
            rand_pts.sort(key=lambda t: t[0])
            rx = [p[0] for p in rand_pts]; ry = [p[1] for p in rand_pts]
            ax.plot(rx, ry, "--s", color="black", lw=1.2, ms=4, alpha=0.85,
                    zorder=3, label="random sweep (baseline)")
        for feat, color, marker in FOCUS_FEATS:
            pts = []
            for name, s in overall.items():
                parsed = _parse_rule_name(name)
                if parsed is None: continue
                f_, tt = parsed
                if f_ != feat: continue
                pts.append((_x_value(s, xkey), s["mAP@50"], tt))
            if not pts: continue
            pts.sort(key=lambda t: t[0])
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            tts = [p[2] for p in pts]
            ax.plot(xs, ys, "-", color=color, lw=1.6, marker=marker, ms=5,
                    alpha=0.9, label=f"{feat} (τ sweep)")
            # mark each τ along the curve, but only on the %super panel
            if xkey == "pct_super":
                for xv, yv, tt in zip(xs, ys, tts):
                    ax.annotate(f"{tt/100:.2f}", (xv, yv), fontsize=6,
                                xytext=(3, 3), textcoords="offset points",
                                color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP@50 (accumulated)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax_p.set_xlim(-5, 105)
    ax_l.set_title("trade-off vs latency", fontsize=11)
    ax_p.set_title("trade-off vs %super-net used", fontsize=11)
    fig.suptitle("KITTI Tracking — confidence-based routing vs always-super/always-base",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(project / "trade_off_focus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] figure -> {project / 'trade_off_focus.png'}")

    # Markdown table
    rows = []
    for n, s in summary["overall"].items():
        rows.append({
            "strategy":   n,
            "mAP@50":     f"{s['mAP@50']:.4f}",
            "mAP@.5:.95": f"{s['mAP@50:95']:.4f}",
            "lat_ms":     f"{s['mean_latency_ms']:.2f}",
            "FPS":        f"{s['mean_fps']:.1f}",
            "GFLOPs":     f"{s['mean_GFLOPs']:.2f}",
            "mJ":         f"{s['mean_energy_mJ']:.2f}",
            "%base":      f"{s['pct_base']:.1f}",
        })
    cols = list(rows[0].keys())
    md_path = project / "per_strategy_summary.md"
    widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |\n")
        f.write("|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r[c].ljust(widths[c]) for c in cols) + " |\n")
    print(f"[*] table -> {md_path}")


if __name__ == "__main__":
    main()
