"""Unified video evaluation for KITTI / BDD100K / Waymo.

Recursive temporal routing: the path used for frame t is decided from frame
t-1's chosen-path signal. Every scored frame runs BOTH BASE and SUPER so all
strategies share one detection pool.

Usage:
    # KITTI
    python -m method_advantage_regress.eval.eval_video \
        --dataset kitti --weight finetuning_AnyDepthYOLO/weights/kitti/best.pt \
        --policies "s0=.../router_s0.pt,s1=.../router_s1.pt" \
        --val_cache method_advantage_regress/outputs/kitti/cache_val_g2.pt \
        --imgsz 384 1248 --conf 0.25 --budgets 10,20,30,40,50,60,70,80,90 \
        --out method_advantage_regress/outputs/kitti/eval/video_curve.json

    # BDD100K
    python -m method_advantage_regress.eval.eval_video \
        --dataset bdd100k --weight finetuning_AnyDepthYOLO/weights/bdd100k/best.pt \
        --policies "s0=.../router_s0.pt,s1=.../router_s1.pt" \
        --val_cache method_advantage_regress/outputs/bdd100k/cache_val_both.pt \
        --imgsz 720 1280 --conf 0.25 --budgets 10,20,30,40,50,60,70,80,90 \
        --out method_advantage_regress/outputs/bdd100k/eval/video_curve.json

    # Waymo (4 shards across 2 GPUs — see eval/run_waymo_eval_both_robust.sh)
    python -m method_advantage_regress.eval.eval_video \
        --dataset waymo --weight finetuning_AnyDepthYOLO/weights/waymo/best.pt \
        --policies "s0=.../router_s0.pt,..." \
        --val_cache method_advantage_regress/outputs/waymo/cache_val_both.pt \
        --imgsz 1280 1920 --conf 0.25 --pi --router_only \
        --num_shards 4 --shard_id 0 \
        --raw_out method_advantage_regress/outputs/waymo/eval/shard_0.pt
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from method_advantage_regress.router.feature_tap import (
    INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS)
from method_advantage_regress.router.router_net import RouterNetwork

# ── BDD100K class mapping ─────────────────────────────────────────────────────
_MOT_TO_ID = {
    "pedestrian": 0, "rider": 1, "car": 2, "bus": 3, "truck": 4,
    "bicycle": 5, "motorcycle": 6, "train": 9,
}
BDD_EVAL_CLS = sorted(set(_MOT_TO_ID.values()))
BDD_MOT_EVAL_CLS = BDD_EVAL_CLS   # alias for backward compatibility

# ── Waymo class mapping ───────────────────────────────────────────────────────
WAYMO_EVAL_CLS = [0, 1, 2]   # vehicle, pedestrian, cyclist


# ── dataset-specific: label parsing & frame iteration ────────────────────────

def _kitti_segments(args):
    label_dir = Path(args.data_root) / "training" / "label_02"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.txt"))
    return seqs


def _kitti_frames(seq, args):
    """Yield (frame_idx, bgr) for a KITTI sequence."""
    img_dir = Path(args.data_root) / "training" / "image_02" / seq
    paths = sorted(img_dir.glob("*.png")) or sorted(img_dir.glob("*.jpg"))
    if args.limit > 0:
        paths = paths[:args.limit]
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is not None:
            yield int(p.stem), bgr


def _kitti_gt(seq, args):
    lpath = Path(args.data_root) / "training" / "label_02" / f"{seq}.txt"
    gt, dc = B.parse_kitti_labels(lpath)
    return gt, dc


def _bdd_segments(args):
    label_dir = Path(args.data_root) / "labels"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.json"))
    if args.limit > 0:
        seqs = seqs[:args.limit]
    return seqs


def _bdd_frames(seq, args):
    """Yield (frame_idx, bgr) for a BDD100K MOT sequence (.mov)."""
    label_dir = Path(args.data_root) / "labels"
    video_dir = Path(args.data_root) / "videos"
    gt = _bdd_parse_labels(label_dir / f"{seq}.json")
    yield from _bdd_labeled_frames(video_dir / f"{seq}.mov", gt)


def _bdd_parse_labels(json_path):
    gt = {}
    for fr in json.loads(Path(json_path).read_text()):
        fi = int(fr["frameIndex"])
        boxes = []
        for l in fr.get("labels", []):
            cid = _MOT_TO_ID.get(l.get("category"))
            b = l.get("box2d")
            if cid is None or b is None:
                continue
            boxes.append((cid, float(b["x1"]), float(b["y1"]),
                          float(b["x2"]), float(b["y2"])))
        gt[fi] = boxes
    return gt


_ANGLE_TO_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _bdd_labeled_frames(mov_path, gt, fps_label=5.0):
    cap = cv2.VideoCapture(str(mov_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    meta = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = fps / fps_label
    targets = {max(0, int(round(fi * step)) - int(round(step))): fi for fi in sorted(gt)}
    di, got, need = 0, 0, len(targets)
    while got < need:
        ok, frame = cap.read()
        if not ok:
            break
        if di in targets:
            if frame.shape[0] > frame.shape[1] and meta in _ANGLE_TO_ROT:
                frame = cv2.rotate(frame, _ANGLE_TO_ROT[meta])
            yield targets[di], frame
            got += 1
        di += 1
    cap.release()


def _bdd_gt(seq, args):
    label_dir = Path(args.data_root) / "labels"
    gt = _bdd_parse_labels(label_dir / f"{seq}.json")
    return gt, {}   # no dontcare in BDD


def _waymo_segments(args):
    label_dir = Path(args.data_root) / "labels"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.txt"))
    if args.limit > 0:
        seqs = seqs[:args.limit]
    return seqs


def _waymo_frames(seq, args):
    """Yield (frame_idx, bgr) for a Waymo segment (pre-extracted JPEGs)."""
    label_dir = Path(args.data_root) / "labels"
    img_root = Path(args.data_root) / "images"
    gt_norm = _waymo_parse_labels(label_dir / f"{seq}.txt")
    for fi in sorted(gt_norm):
        p = img_root / seq / f"{fi:06d}.jpg"
        if not p.exists():
            continue
        bgr = cv2.imread(str(p))
        if bgr is not None:
            yield fi, bgr


def _waymo_parse_labels(txt_path):
    gt = {}
    if not Path(txt_path).exists():
        return gt
    for ln in Path(txt_path).read_text().splitlines():
        f = ln.split()
        if len(f) != 6:
            continue
        fi = int(f[0]); cls = int(f[1])
        xc, yc, w, h = (float(x) for x in f[2:])
        gt.setdefault(fi, []).append((cls, xc, yc, w, h))
    return gt


def _waymo_gt(seq, args):
    label_dir = Path(args.data_root) / "labels"
    return _waymo_parse_labels(label_dir / f"{seq}.txt"), {}


def _waymo_gt_pixel(fidx, gt_norm, H, W):
    boxes = []
    for cls, xc, yc, w, h in gt_norm.get(fidx, []):
        x1 = (xc - w / 2) * W; y1 = (yc - h / 2) * H
        x2 = (xc + w / 2) * W; y2 = (yc + h / 2) * H
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


# ── shared utilities ──────────────────────────────────────────────────────────

def grid_vec(captured, layers, G):
    return torch.cat(
        [F.adaptive_avg_pool2d(captured[i].float(), G).squeeze(0) for i in layers], dim=0)


def pixel_signals(bgr):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:, :, 0]
    lum = float(y.mean())
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.sqrt(gx * gx + gy * gy).mean())
    return lum, edge


def load_router(path, device):
    from method_advantage_regress.router.router_net import GapMlpNet
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    sd = ckpt["state_dict"]
    is_gap = not any(k.endswith("weight") and v.dim() == 4 for k, v in sd.items())
    cls = GapMlpNet if is_gap else RouterNetwork
    net = cls(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
              hidden_dim=a.get("hidden", 64), feat=a.get("feat", "both"),
              norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(device)
    return net, ckpt, a.get("feat", "both"), is_gap


class PIController:
    def __init__(self, target, kp, ki, beta, tau0, tau_lo=-2.0, tau_hi=2.0):
        self.Lstar, self.kp, self.ki, self.beta, self.tau0 = target, kp, ki, beta, tau0
        self.tau_lo, self.tau_hi = tau_lo, tau_hi
        self.reset()

    def reset(self):
        self.I = 0.0; self.Lbar = 0.5; self.tau = self.tau0

    def __call__(self, prev_choice, prev_value, frame_idx):
        choice = "super" if (frame_idx > 0 and prev_value > self.tau) else "base"
        ell = 1.0 if choice == "super" else 0.0
        self.Lbar = self.beta * self.Lbar + (1.0 - self.beta) * ell
        e = self.Lstar - self.Lbar
        self.I += e
        self.tau = min(max(self.tau0 - self.kp * e - self.ki * self.I,
                           self.tau_lo), self.tau_hi)
        return choice


# ── main ──────────────────────────────────────────────────────────────────────

# public aliases kept for backward compatibility with jetson scripts
parse_box_track = _bdd_parse_labels
labeled_frames  = _bdd_labeled_frames


def main():
    import method_advantage_regress.eval.eval_utils as B  # lazy: pyc not needed for TRT demo
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bdd100k",
                    choices=["kitti", "bdd100k", "waymo"])
    ap.add_argument("--weight", required=True)
    ap.add_argument("--policies", default="",
                    help="comma-sep tag=path, e.g. s0=...pt,s1=...pt")
    ap.add_argument("--router", default=None, help="single router shorthand")
    ap.add_argument("--data_root", default=None,
                    help="dataset root (default: /media/data/<dataset>_tracking or _mot/val)")
    ap.add_argument("--sequences", type=str, nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, nargs=2, default=None,
                    help="inference size HxW (default: dataset preset)")
    ap.add_argument("--grid", default="2",
                    help="spatial grid G or HxW; must match build_cache --grid")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--router_only", action="store_true")
    ap.add_argument("--no_conf", action="store_true")
    ap.add_argument("--router_taus", type=int, default=21)
    ap.add_argument("--val_cache", default=None)
    ap.add_argument("--budgets", default="10,20,30,40,50,60,70,80,90")
    ap.add_argument("--pi", action="store_true")
    ap.add_argument("--pi_targets", default=None)
    ap.add_argument("--pi_kp", type=float, default=2.0)
    ap.add_argument("--pi_ki", type=float, default=0.2)
    ap.add_argument("--pi_beta", type=float, default=0.9)
    ap.add_argument("--pi_tau0", type=float, default=0.0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--raw_out", default=None,
                    help="dump raw shard matches (merge with merge_video_shards.py)")
    ap.add_argument("--thresh_clips", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ── defaults per dataset ──────────────────────────────────────────────────
    DATA_ROOTS = {
        "kitti":   "/media/data/kitti-tracking",
        "bdd100k": "/media/data/bdd100k_mot/val",
        "waymo":   "/media/data/waymo/val",
    }
    IMGSZ = {"kitti": [384, 1248], "bdd100k": [720, 1280], "waymo": [1280, 1920]}
    if args.data_root is None:
        args.data_root = DATA_ROOTS[args.dataset]
    if args.imgsz is None:
        args.imgsz = IMGSZ[args.dataset]
    args.grid = (lambda s: tuple(int(x) for x in s.split("x")) if "x" in s
                 else (int(s), int(s)))(str(args.grid))
    _base = Path(__file__).resolve().parent.parent / "outputs" / args.dataset
    if args.out is None:
        args.out = str(_base / "eval" / "video_curve.json")

    # ── dataset-specific dispatch ─────────────────────────────────────────────
    if args.dataset == "kitti":
        B.EVAL_CLS = list(range(8))          # KITTI: all YOLO classes
        get_segs   = _kitti_segments
        get_frames = _kitti_frames
        get_gt     = _kitti_gt
    elif args.dataset == "bdd100k":
        B.EVAL_CLS = BDD_EVAL_CLS
        get_segs   = _bdd_segments
        get_frames = _bdd_frames
        get_gt     = _bdd_gt
    else:   # waymo
        B.EVAL_CLS = WAYMO_EVAL_CLS
        get_segs   = _waymo_segments
        get_frames = _waymo_frames
        get_gt     = _waymo_gt

    device = args.device if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.weight, task="detect")
    yolo.model.to(device).eval()
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super, skip_base = [False] * N, [True] * N
    gs, gb = B.measure_flops_super_base(yolo.model, args.imgsz, device)
    print(f"[*] GFLOPs super={gs:.2f} base={gb:.2f}  EVAL_CLS={B.EVAL_CLS}")

    # ── load policies ─────────────────────────────────────────────────────────
    if args.policies:
        pol_specs = [kv.split("=", 1) for kv in args.policies.split(",")]
    elif args.router:
        pol_specs = [("router", args.router)]
    else:
        pol_specs = []
    nets, feats, gaps = {}, {}, {}
    dummy_in = torch.zeros(2, 768, args.grid[0], args.grid[1], device=device)
    dummy_pr = torch.zeros(2, 640, args.grid[0], args.grid[1], device=device)
    zpid = torch.zeros(2, dtype=torch.long, device=device)
    for tag, path in pol_specs:
        net, ckpt, feat, is_gap = load_router(path, device)
        net.eval()
        di = dummy_in.mean(dim=(2, 3)) if is_gap else dummy_in
        dp = dummy_pr.mean(dim=(2, 3)) if is_gap else dummy_pr
        with torch.no_grad():
            net(di, None if feat == "input" else dp, zpid)
        net.load_state_dict(ckpt["state_dict"]); net.eval()
        nets[tag] = net; feats[tag] = feat; gaps[tag] = is_gap
    need_pred = any(f != "input" for f in feats.values())
    print(f"[*] policies: {list(nets)} (need_pred={need_pred})")

    # ── hooks ─────────────────────────────────────────────────────────────────
    captured = {}
    for idx in STATE_LAYERS:
        yolo.model.model[idx].register_forward_hook(
            lambda m, i, o, k=idx: captured.__setitem__(k, o))

    # ── sequences ─────────────────────────────────────────────────────────────
    seqs = get_segs(args)
    eval_seqs = seqs[args.shard_id::args.num_shards] if args.num_shards > 1 else seqs

    # ── lum/edge percentile thresholds ────────────────────────────────────────
    lum_taus = edge_taus = None
    pcts = list(range(5, 100, 5))
    if not args.router_only:
        lum_all, edge_all = [], []
        thr_seqs = seqs if args.thresh_clips <= 0 else seqs[:args.thresh_clips]
        for seq in thr_seqs:
            for _, bgr in get_frames(seq, args):
                l, e = pixel_signals(bgr); lum_all.append(l); edge_all.append(e)
        if lum_all:
            lum_taus = np.percentile(lum_all, pcts)
            edge_taus = np.percentile(edge_all, pcts)
            print(f"[*] lum/edge thresholds from {len(lum_all)} frames ({len(thr_seqs)} clips)")

    # ── val-derived router thresholds ─────────────────────────────────────────
    val_taus = None
    if args.val_cache:
        budgets = [int(b) for b in args.budgets.split(",")]
        vc = torch.load(args.val_cache, map_location="cpu", weights_only=False)
        v_in = vc["input_base"].to(device)
        v_pr = vc["pred_base"].to(device) if "pred_base" in vc else None
        v_pid = torch.zeros(v_in.shape[0], dtype=torch.long, device=device)
        val_taus = {}
        with torch.no_grad():
            for tag, net in nets.items():
                xi = v_in.mean(dim=(2, 3)) if gaps[tag] else v_in
                pr = None if feats[tag] == "input" else (
                    v_pr.mean(dim=(2, 3)) if gaps[tag] else v_pr)
                ah = net.logit(xi, pr, v_pid).view(-1).cpu().numpy()
                val_taus[tag] = {b: float(np.quantile(ah, 1.0 - b / 100.0)) for b in budgets}
        print(f"[*] val thresholds (budgets={budgets})")

    # ── strategies ────────────────────────────────────────────────────────────
    strategies = [
        dict(name="always_base",  kind="const", thres=0,
             decide=lambda pc, pv, fi: "base"),
        dict(name="always_super", kind="const", thres=0,
             decide=lambda pc, pv, fi: "super"),
    ]
    if not args.router_only and lum_taus is not None:
        for p in range(0, 101, 10):
            ps = p / 100.0
            strategies.append(dict(
                name=f"random_p{p:03d}", kind="random", thres=ps,
                decide=lambda pc, pv, fi, ps=ps: "super" if np.random.random() < ps else "base"))
        for kind, taus in (("lum", lum_taus), ("edge", edge_taus)):
            for pc, tau in zip(pcts, taus):
                strategies.append(dict(
                    name=f"{kind}_p{pc:02d}", kind=kind, thres=float(tau),
                    decide=lambda pc_, pv, fi, tau=tau: "super" if pv < tau else "base"))
        if not args.no_conf:
            conf_taus = [round((i + 1) / 10.0, 2) for i in range(9)]
            for kind in ("conftop20",):
                for tau in conf_taus:
                    strategies.append(dict(
                        name=f"{kind}_t{int(tau*100):02d}", kind=kind, thres=tau,
                        decide=lambda pc, pv, fi, tau=tau: "base" if (fi == 0 or pv >= tau) else "super"))

    policy_taus = [round(-0.4 + (2.0 / (args.router_taus - 1)) * i, 3)
                   for i in range(args.router_taus)]
    for tag in nets:
        pname = f"policy_{tag}"
        if val_taus is not None:
            for b, tau in val_taus[tag].items():
                strategies.append(dict(
                    name=f"{pname}_b{b:02d}", kind="router", ptag=tag,
                    thres=tau, budget=b,
                    decide=lambda pc, pv, fi, tau=tau: "super" if (fi > 0 and pv > tau) else "base"))
        else:
            for tau in policy_taus:
                strategies.append(dict(
                    name=f"{pname}_t{int(tau*100):+04d}", kind="router", ptag=tag,
                    thres=tau,
                    decide=lambda pc, pv, fi, tau=tau: "super" if (fi > 0 and pv > tau) else "base"))
        if args.pi:
            pi_targets = [int(t) for t in (args.pi_targets or args.budgets).split(",")]
            for t in pi_targets:
                ctrl = PIController(target=t / 100.0, kp=args.pi_kp, ki=args.pi_ki,
                                    beta=args.pi_beta, tau0=args.pi_tau0)
                strategies.append(dict(
                    name=f"{pname}_pi{t:02d}", kind="router", ptag=tag,
                    thres=args.pi_tau0, budget=t, ctrl=ctrl, decide=ctrl))

    print(f"[*] shard {args.shard_id}/{args.num_shards}: "
          f"{len(eval_seqs)}/{len(seqs)} seqs, {len(strategies)} strategies")

    # ── eval loop ─────────────────────────────────────────────────────────────
    np.random.seed(0)
    state = {s["name"]: B.StrategyState(s["name"]) for s in strategies}
    one  = torch.ones(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, dtype=torch.long, device=device)
    total_frames = 0

    for vi, seq in enumerate(eval_seqs):
        gt_raw, dc_raw = get_gt(seq, args)
        for s in state.values():
            s.prev_choice = None
        for st in strategies:
            if "ctrl" in st:
                st["ctrl"].reset()
        nfr = 0

        for fi_pos, (fidx, bgr) in enumerate(get_frames(seq, args)):
            total_frames += 1; nfr += 1
            H, W = bgr.shape[:2]
            lum_cur, edge_cur = pixel_signals(bgr)

            captured.clear()
            r_super = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                                   iou=0.7, skip=skip_super, verbose=False, device=device)[0]
            in_s = grid_vec(captured, INPUT_LEVEL_LAYERS, args.grid).unsqueeze(0)
            pr_s = grid_vec(captured, PRED_LEVEL_LAYERS, args.grid).unsqueeze(0) if need_pred else None
            captured.clear()
            r_base = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                                  iou=0.7, skip=skip_base, verbose=False, device=device)[0]
            in_b = grid_vec(captured, INPUT_LEVEL_LAYERS, args.grid).unsqueeze(0)
            pr_b = grid_vec(captured, PRED_LEVEL_LAYERS, args.grid).unsqueeze(0) if need_pred else None

            preds = {"super": B.boxes_to_preds(r_super), "base": B.boxes_to_preds(r_base)}
            conf_s = r_super.boxes.conf.cpu() if (r_super.boxes is not None and len(r_super.boxes)) else torch.empty(0)
            conf_b = r_base.boxes.conf.cpu() if (r_base.boxes is not None and len(r_base.boxes)) else torch.empty(0)

            def sig(conf):
                return {"conftop20": B.conf_top_k_mean(conf, 20) if conf.numel() else 0.0,
                        "confge10":  B.conf_mean_ge(conf, 0.1)   if conf.numel() else 0.0}
            sig_super, sig_base = sig(conf_s), sig(conf_b)

            with torch.no_grad():
                av = {}
                for tag, n in nets.items():
                    xs_i, xb_i = ((in_s.mean(dim=(2, 3)), in_b.mean(dim=(2, 3)))
                                  if gaps[tag] else (in_s, in_b))
                    if feats[tag] == "input":
                        ps_i = pb_i = None
                    else:
                        ps_i, pb_i = ((pr_s.mean(dim=(2, 3)), pr_b.mean(dim=(2, 3)))
                                      if gaps[tag] else (pr_s, pr_b))
                    av[tag] = {"super": float(n.logit(xs_i, ps_i, one)),
                               "base":  float(n.logit(xb_i, pb_i, zero))}

            # ground-truth boxes in pixel coords
            if args.dataset == "waymo":
                gts = _waymo_gt_pixel(fidx, gt_raw, H, W)
                dc  = []
            else:
                gts = gt_raw.get(fidx, [])
                dc  = dc_raw.get(fidx, []) if dc_raw else []

            for st in strategies:
                kind, decide = st["kind"], st["decide"]
                s = state[st["name"]]
                if kind in ("conftop20", "confge10"):
                    pv = ((sig_super[kind] if s.prev_choice == "super" else sig_base[kind])
                          if s.prev_choice else 0.0)
                elif kind == "router":
                    avt = av[st["ptag"]]
                    pv = ((avt["super"] if s.prev_choice == "super" else avt["base"])
                          if s.prev_choice else 0.0)
                elif kind == "lum":
                    pv = lum_cur
                elif kind == "edge":
                    pv = edge_cur
                else:
                    pv = 0.0
                choice = decide(s.prev_choice, pv, fi_pos)
                s.prev_choice = choice
                s.n_super += (choice == "super"); s.n_base += (choice == "base")
                p = B.filter_dontcare(preds[choice], dc) if dc else preds[choice]
                m, gtc = B.match_frame_multi_iou(p, gts)
                s.matches_multi.extend(m)
                for cls_id, cnt in gtc.items():
                    s.gt_count[cls_id] += cnt

        print(f"[{vi+1}/{len(eval_seqs)} {seq}] {nfr} labeled frames")

    # ── shard dump ────────────────────────────────────────────────────────────
    if args.raw_out:
        raw = {"gflops_super": gs, "gflops_base": gb, "total_frames": total_frames,
               "meta": {s["name"]: {"kind": s["kind"], "thres": s["thres"],
                                    "budget": s.get("budget")} for s in strategies},
               "state": {s["name"]: {"matches_multi": state[s["name"]].matches_multi,
                                     "gt_count": dict(state[s["name"]].gt_count),
                                     "n_super": state[s["name"]].n_super,
                                     "n_base":  state[s["name"]].n_base}
                         for s in strategies}}
        rp = Path(args.raw_out); rp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw, rp)
        print(f"[*] shard {args.shard_id} raw -> {rp}")
        return

    # ── aggregate ─────────────────────────────────────────────────────────────
    rows = []
    for st in strategies:
        s = state[st["name"]]
        _, map50, map5095 = B.dataset_map_multi_iou(s.matches_multi, s.gt_count)
        n = s.n_super + s.n_base
        super_rate = s.n_super / max(n, 1)
        gflops = super_rate * gs + (1 - super_rate) * gb
        nm = st["name"]
        if "_pi" in nm:
            family = nm.rsplit("_pi", 1)[0] + "_pi"
        elif "_t" in nm:
            family = nm.rsplit("_t", 1)[0]
        elif "_b" in nm:
            family = nm.rsplit("_b", 1)[0]
        else:
            family = st["kind"]
        rows.append({"name": nm, "kind": st["kind"], "family": family,
                     "thres": st["thres"], "budget": st.get("budget"),
                     "map50": map50, "map": map5095,
                     "super_rate": super_rate, "gflops": gflops})
    rows.sort(key=lambda r: r["gflops"])

    hdr = f"{'strategy':<26}{'super%':>8}{'GFLOPs':>9}{'mAP50':>9}{'mAP':>9}"
    lines = [hdr] + [
        f"{r['name']:<26}{r['super_rate']*100:>7.1f}%{r['gflops']:>9.2f}"
        f"{r['map50']:>9.4f}{r['map']:>9.4f}" for r in rows]
    print("\n" + "\n".join(lines))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gflops_super": gs, "gflops_base": gb, "rows": rows}, indent=2))
    with open(out.with_suffix(".log"), "w") as f:
        f.write(f"# eval_video.py {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# args: {json.dumps(vars(args))}\n")
        f.write(f"# seqs={len(seqs)} frames={total_frames} "
                f"strategies={len(strategies)} base={gb:.2f} super={gs:.2f}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"[*] saved -> {out}\n[*] log -> {out.with_suffix('.log')}")


if __name__ == "__main__":
    main()
