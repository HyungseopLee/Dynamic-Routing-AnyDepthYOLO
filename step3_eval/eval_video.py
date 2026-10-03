"""Unified video evaluation for KITTI / BDD100K / Waymo.

Recursive temporal routing: the path used for frame t is decided from frame
t-1's chosen-path signal (causal -- no future information). Every scored frame
runs BOTH BASE and SUPER so all strategies share one detection pool.

Pass one --policies entry per seed to reproduce the paper's seed-averaged curves;
a single seed is shown below for brevity. Defaults match the paper: --grid 2 over
feat=both routers.

Usage:
The three differ only in --dataset, --imgsz and the cache/weight paths.

    # KITTI
    python -m step3_eval.eval_video \
        --dataset kitti --weight results/step1_finetune/weights/kitti/best.pt \
        --policies  "s0=results/step2_router/weights/kitti/router_g2x2_both_s0.pt" \
        --val_cache results/step2_router/cache/kitti/cache_val_g2x2_both.pt \
        --grid 2 --imgsz 384 1248 --conf 0.25 \
        --budgets 10,20,30,40,50,60,70,80,90 \
        --out results/step3_eval/kitti/eval/video_curve.json

    # BDD100K
    python -m step3_eval.eval_video \
        --dataset bdd100k --weight results/step1_finetune/weights/bdd100k/best.pt \
        --policies  "s0=results/step2_router/weights/bdd100k/router_g2x2_both_s0.pt" \
        --val_cache results/step2_router/cache/bdd100k/cache_val_g2x2_both.pt \
        --grid 2 --imgsz 720 1280 --conf 0.25 \
        --budgets 10,20,30,40,50,60,70,80,90 \
        --out results/step3_eval/bdd100k/eval/video_curve.json

    # Waymo
    python -m step3_eval.eval_video \
        --dataset waymo --weight results/step1_finetune/weights/waymo/best.pt \
        --policies  "s0=results/step2_router/weights/waymo/router_g2x2_both_s3.pt" \
        --val_cache results/step2_router/cache/waymo/cache_val_g2x2_both.pt \
        --grid 2 --imgsz 1280 1920 --conf 0.25 \
        --budgets 10,20,30,40,50,60,70,80,90 \
        --out results/step3_eval/waymo/eval/video_curve.json

Waymo at 1280x1920 is heavy enough to want sharding: add --num_shards N --shard_id I
with --raw_out results/step3_eval/waymo/eval/shard_<I>.pt, then merge the shards with
step3_eval/merge_video_shards.py (see step3_eval/run_waymo_eval_both_robust.sh).
"""
import argparse
import json
import gc
import sys
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from router.feature_tap import (
    INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS)
from router.router_net import RouterNetwork

# ── BDD100K class mapping ─────────────────────────────────────────────────────
_MOT_TO_ID = {
    "pedestrian": 0, "rider": 1, "car": 2, "bus": 3, "truck": 4,
    "bicycle": 5, "motorcycle": 6, "train": 9,
}
BDD_EVAL_CLS = sorted(set(_MOT_TO_ID.values()))
BDD_MOT_EVAL_CLS = BDD_EVAL_CLS   # alias for backward compatibility

# ── Waymo class mapping ───────────────────────────────────────────────────────
WAYMO_EVAL_CLS = [0, 1, 2]   # vehicle, pedestrian, cyclist

# class id -> name, per dataset (only used to label the COCO dump's categories)
CLASS_NAMES = {
    "kitti":   {0: "car", 1: "van", 2: "truck", 3: "pedestrian", 4: "person_sitting",
                5: "cyclist", 6: "tram", 7: "misc"},
    "bdd100k": {v: k for k, v in _MOT_TO_ID.items()},
    "waymo":   {0: "vehicle", 1: "pedestrian", 2: "cyclist"},
}


# ── dataset-specific: label parsing & frame iteration ────────────────────────

def _kitti_segments(args):
    label_dir = Path(args.data_root) / "training" / "label_02"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.txt"))
    return seqs


def topk_per_class(preds, k):
    """Keep the k highest-scoring detections of each class (COCO maxDets)."""
    if not k or len(preds) <= k:
        return preds
    by_cls = defaultdict(list)
    for p in preds:
        by_cls[p[0]].append(p)
    out = []
    for v in by_cls.values():
        out.extend(sorted(v, key=lambda x: -x[1])[:k] if len(v) > k else v)
    return out


class FrameDump:
    """Per-frame, per-path record of everything a routing policy needs.

    The detector is run on both paths for every frame anyway, so one pass can store
    each path's matched detections plus the router outputs. Scoring a policy then
    reduces to picking, per frame, which path's rows to pool -- no GPU, milliseconds
    per policy. That is what makes an exact tau sweep feasible: the realised SUPER
    usage of a threshold can only be known by replaying the recursion over the whole
    video, which would otherwise cost a full evaluation run per candidate tau.
    """

    def __init__(self, tags):
        self.tags = list(tags)
        self.rows = []          # one dict per frame
        self.gt = []            # per-frame {cls: count}

    def add(self, seq, fidx, gts, dc, preds, av, B, max_det, topk):
        rec = {"seq": seq, "frame": int(fidx)}
        for pth in ("base", "super"):
            pp = B.filter_dontcare(preds[pth], dc) if dc else preds[pth]
            m, gtc = B.match_frame_multi_iou(topk(pp, max_det), gts)
            rec[f"cls_{pth}"] = np.fromiter((x[0] for x in m), np.int16, len(m))
            rec[f"conf_{pth}"] = np.fromiter((x[1] for x in m), np.float32, len(m))
            # a frame can have zero detections; np.asarray([]) is 1-D, and
            # reshape(0, -1) cannot infer the column count from an empty array
            flags = [x[2] for x in m]
            rec[f"tp_{pth}"] = (np.asarray(flags, bool) if flags
                                else np.empty((0, len(B.IOU_GRID)), bool))
        self.gt.append(gtc)
        for t in self.tags:
            rec[f"ah_{t}_base"] = av[t]["base"]
            rec[f"ah_{t}_super"] = av[t]["super"]
        self.rows.append(rec)

    def save(self, path, meta):
        out = {"seq": np.array([r["seq"] for r in self.rows]),
               "frame": np.array([r["frame"] for r in self.rows], np.int32)}
        for pth in ("base", "super"):     # not `path`: that is this method's argument
            out[f"n_{pth}"] = np.array([len(r[f"cls_{pth}"]) for r in self.rows], np.int32)
            out[f"cls_{pth}"] = np.concatenate([r[f"cls_{pth}"] for r in self.rows])
            out[f"conf_{pth}"] = np.concatenate([r[f"conf_{pth}"] for r in self.rows])
            tp = [r[f"tp_{pth}"] for r in self.rows]
            n_iou = max((t.shape[1] for t in tp if t.size), default=10)
            out[f"tp_{pth}"] = np.concatenate(
                [t if t.size else np.empty((0, n_iou), bool) for t in tp])
        for t in self.tags:
            out[f"ah_{t}_base"] = np.array([r[f"ah_{t}_base"] for r in self.rows], np.float32)
            out[f"ah_{t}_super"] = np.array([r[f"ah_{t}_super"] for r in self.rows], np.float32)
        cls_ids = sorted({c for g in self.gt for c in g})
        out["gt_cls"] = np.array(cls_ids, np.int16)
        out["gt_count"] = np.array([[g.get(c, 0) for c in cls_ids] for g in self.gt], np.int32)
        out["meta"] = np.array([json.dumps(meta)])
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **out)
        print(f"[*] frame dump ({len(self.rows)} frames) -> {p}")


def _kitti_frames(seq, args):
    """Yield (frame_idx, bgr) for a KITTI sequence.

    With --shuffle_frames the frames are emitted in random order. Each frame is
    still scored against its own ground truth (the yielded index is the true frame
    index), so only temporal coherence is destroyed: the routing signal carried
    from the previous decision now comes from an unrelated scene. That is the
    worst case for recursive routing -- every frame is a hard scene cut.
    """
    img_dir = Path(args.data_root) / "training" / "image_02" / seq
    paths = sorted(img_dir.glob("*.png")) or sorted(img_dir.glob("*.jpg"))
    if args.limit > 0:
        paths = paths[:args.limit]
    if args.shuffle_frames:
        # crc32, not hash(): python string hashing is salted per process
        rng = np.random.default_rng(zlib.crc32(f"{seq}:{args.shuffle_seed}".encode()))
        paths = [paths[i] for i in rng.permutation(len(paths))]
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is not None:
            yield int(p.stem), bgr


def _kitti_gt(seq, args):
    import step3_eval.eval_utils as _B
    lpath = Path(args.data_root) / "training" / "label_02" / f"{seq}.txt"
    gt, dc = _B.parse_kitti_labels(lpath)
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


def load_router(path, device):
    from router.router_net import GapMlpNet
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    sd = ckpt["state_dict"]
    is_gap = not any(k.endswith("weight") and v.dim() == 4 for k, v in sd.items())
    cls = GapMlpNet if is_gap else RouterNetwork
    net = cls(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
              hidden_dim=a.get("hidden", 64), feat=a.get("feat", "both"),
              norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(device)
    # remap legacy GapMlpNet keys: input_proj.0/* -> input_proj.norm/fc
    if is_gap and any(k.startswith("input_proj.0.") for k in sd):
        remap = {}
        for k, v in sd.items():
            nk = k.replace("input_proj.0.", "input_proj.norm.") \
                   .replace("input_proj.1.", "input_proj.fc.") \
                   .replace("pred_proj.0.",  "pred_proj.norm.") \
                   .replace("pred_proj.1.",  "pred_proj.fc.")
            remap[nk] = v
        sd = remap
        ckpt = dict(ckpt, state_dict=sd)
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
    import step3_eval.eval_utils as B  # lazy: pyc not needed for TRT demo
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
                    help="spatial grid G or HxW; must match build_cache --grid "
                         "(paper default: 2)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--router_only", action="store_true")
    ap.add_argument("--router_taus", type=int, default=21)
    ap.add_argument("--val_cache", default=None)
    ap.add_argument("--budgets", default="10,20,30,40,50,60,70,80,90")
    ap.add_argument("--pi", action="store_true")
    ap.add_argument("--pi_targets", default=None)
    ap.add_argument("--pi_kp", type=float, default=2.0)
    ap.add_argument("--pi_ki", type=float, default=0.2)
    ap.add_argument("--pi_beta", type=float, default=0.9)
    ap.add_argument("--pi_tau0", type=float, default=0.0)
    ap.add_argument("--max_det", type=int, default=100,
                    help="COCO maxDets: keep the top-N detections per (frame, class) "
                         "before matching. pycocotools applies its maxDets per image AND "
                         "per category, so this reproduces the standard AP/AR@100. "
                         "0 disables the cap (the pre-2026-09 behaviour, AR@inf)")
    ap.add_argument("--coco_out", default=None,
                    help="directory for a COCO-format dump + pycocotools summarize() of "
                         "the strategies named by --coco_strategies")
    ap.add_argument("--coco_strategies", default="",
                    help="comma-sep strategy names to dump for pycocotools, e.g. "
                         "'always_base,always_super,policy_s0_b50'. Empty = the two const "
                         "strategies plus every router strategy of the first policy")
    ap.add_argument("--frame_dump", default=None,
                    help="write per-frame, per-path detection matches and router outputs "
                         "to this .npz. Any per-frame path-selection policy can then be "
                         "scored offline without touching the GPU, which is what "
                         "step3_eval/ablation/tau_sweep.py uses to solve for the tau that "
                         "realises an exact SUPER usage (10%, 20%, ...)")
    ap.add_argument("--causal", action="store_true",
                    help="decide frame t from the A-hat produced while executing frame t-1, "
                         "matching the deployed controller. Default (off) decides frame t "
                         "from A-hat computed on frame t itself")
    ap.add_argument("--random_seeds", type=int, default=5,
                    help="independent seeds for the random baseline, so it gets the same "
                         "mean+/-std treatment as the router (was a single realisation)")
    ap.add_argument("--shuffle_frames", action="store_true",
                    help="emit each sequence's frames in random order, destroying temporal "
                         "coherence (KITTI only). Frames are still scored against their own "
                         "GT; only the recursive routing signal is invalidated")
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--raw_out", default=None,
                    help="dump raw shard matches (merge with merge_video_shards.py)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.shuffle_frames and args.dataset != "kitti":
        raise SystemExit("--shuffle_frames is implemented for --dataset kitti only")

    # ── defaults per dataset ──────────────────────────────────────────────────
    DATA_ROOTS = {
        "kitti":   "/media/data/kitti-tracking",
        "bdd100k": "/media/data/bdd100k_mot/val",
        "waymo":   "/media/data/waymo_yolo/val",
    }
    IMGSZ = {"kitti": [384, 1248], "bdd100k": [720, 1280], "waymo": [1280, 1920]}
    if args.data_root is None:
        args.data_root = DATA_ROOTS[args.dataset]
    if args.imgsz is None:
        args.imgsz = IMGSZ[args.dataset]
    args.grid = (lambda s: tuple(int(x) for x in s.split("x")) if "x" in s
                 else (int(s), int(s)))(str(args.grid))
    _base = Path(__file__).resolve().parent.parent / "results/step3_eval" / args.dataset
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
    if not args.router_only:
        # One family per seed, mirroring the router: the figure then shows a
        # mean +/- std band for both curves instead of a single noisy realisation
        # against a seed-averaged one. Each family owns its Generator so the draws
        # are reproducible and independent of strategy evaluation order.
        for si in range(args.random_seeds):
            rng = np.random.default_rng(si)      # seeds 0..4, matching the routers
            for p in range(0, 101, 10):
                ps = p / 100.0
                strategies.append(dict(
                    name=f"random_s{si}_p{p:03d}", kind="random", thres=ps,
                    decide=lambda pc, pv, fi, ps=ps, r=rng: (
                        "super" if r.random() < ps else "base")))

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
    if not eval_seqs:
        raise SystemExit(f"no sequences found under --data_root {args.data_root} "
                         f"(dataset={args.dataset}). Without this check the run would "
                         f"'succeed' and write an all-zero curve.")

    # ── optional COCO-format dump for pycocotools cross-check ─────────────────
    dump = None
    if args.coco_out:
        from step3_eval.coco_summary import CocoDump
        if args.coco_strategies:
            want = [n.strip() for n in args.coco_strategies.split(",") if n.strip()]
        else:
            first = next((s["ptag"] for s in strategies if s["kind"] == "router"), None)
            want = [s["name"] for s in strategies
                    if s["kind"] == "const" or s.get("ptag") == first]
        known = {s["name"] for s in strategies}
        missing = [n for n in want if n not in known]
        if missing:
            raise SystemExit(f"--coco_strategies: unknown {missing}; "
                             f"available e.g. {sorted(known)[:6]}")
        names = CLASS_NAMES[args.dataset]
        dump = CocoDump(want, {c: names.get(c, str(c)) for c in B.EVAL_CLS})
        print(f"[*] COCO dump for {len(want)} strategies -> {args.coco_out}")

    # ── eval loop ─────────────────────────────────────────────────────────────
    # The accumulators below are long-lived and cycle-free; with millions of live
    # objects the generational collector costs more than it reclaims. Freeze what
    # already exists and stop automatic passes (memory is still refcount-managed).
    gc.freeze()
    gc.disable()
    np.random.seed(0)
    state = {s["name"]: B.StrategyState(s["name"]) for s in strategies}
    fd = FrameDump(list(nets)) if args.frame_dump else None
    one  = torch.ones(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, dtype=torch.long, device=device)
    total_frames = 0

    for vi, seq in enumerate(eval_seqs):
        gt_raw, dc_raw = get_gt(seq, args)
        for s in state.values():
            s.prev_choice = None
            s.prev_value = 0.0
        for st in strategies:
            if "ctrl" in st:
                st["ctrl"].reset()
        nfr = 0

        for fi_pos, (fidx, bgr) in enumerate(get_frames(seq, args)):
            total_frames += 1; nfr += 1
            H, W = bgr.shape[:2]

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

            iid = dump.add_frame(gts, dc, H, W) if dump else None

            if fd is not None:
                fd.add(seq, fidx, gts, dc, preds, av, B, args.max_det, topk_per_class)

            for st in strategies:
                kind, decide = st["kind"], st["decide"]
                s = state[st["name"]]
                if kind == "router":
                    avt = av[st["ptag"]]
                    if args.causal:
                        # Deployment semantics (online_budget_demo_stream.py): frame t is
                        # executed with the path decided at t-1, and the A-hat it produces
                        # decides frame t+1. The router therefore never sees the frame it
                        # is routing. Without this flag the decision for frame t reads
                        # A-hat computed on frame t itself, which is one frame ahead of
                        # what a causal system can know.
                        pv = s.prev_value
                    else:
                        pv = ((avt["super"] if s.prev_choice == "super" else avt["base"])
                              if s.prev_choice else 0.0)
                else:
                    pv = 0.0
                choice = decide(s.prev_choice, pv, fi_pos)
                if kind == "router":
                    s.prev_value = avt[choice]
                s.prev_choice = choice
                s.n_super += (choice == "super"); s.n_base += (choice == "base")
                p = B.filter_dontcare(preds[choice], dc) if dc else preds[choice]
                p = topk_per_class(p, args.max_det)
                if dump:
                    dump.add_dets(st["name"], iid, p)
                m, gtc = B.match_frame_multi_iou(p, gts)
                s.add_matches(m)
                for cls_id, cnt in gtc.items():
                    s.gt_count[cls_id] += cnt

        for st_ in state.values():
            st_.consolidate()
        print(f"[{vi+1}/{len(eval_seqs)} {seq}] {nfr} labeled frames")

    # ── shard dump ────────────────────────────────────────────────────────────
    if args.raw_out:
        raw = {"gflops_super": gs, "gflops_base": gb, "total_frames": total_frames,
               "meta": {s["name"]: {"kind": s["kind"], "thres": s["thres"],
                                    "budget": s.get("budget")} for s in strategies},
               "compact": True,
               "state": {s["name"]: dict(
                             zip(("cls", "conf", "tp"), state[s["name"]].compact()),
                             gt_count=dict(state[s["name"]].gt_count),
                             n_super=state[s["name"]].n_super,
                             n_base=state[s["name"]].n_base)
                         for s in strategies}}
        rp = Path(args.raw_out); rp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw, rp)
        print(f"[*] shard {args.shard_id} raw -> {rp}")
        return

    # ── aggregate ─────────────────────────────────────────────────────────────
    rows = []
    for st in strategies:
        s = state[st["name"]]
        cls_a, conf_a, tp_a = s.compact()
        _, map50, map5095, ar5095, per_cls = B.dataset_map_ar_compact(
            cls_a, conf_a, tp_a, s.gt_count)
        n = s.n_super + s.n_base
        super_rate = s.n_super / max(n, 1)
        gflops = super_rate * gs + (1 - super_rate) * gb
        nm = st["name"]
        if st["kind"] == "random":
            family = nm.rsplit("_p", 1)[0]        # random_s0_p030 -> random_s0
        elif "_pi" in nm:
            family = nm.rsplit("_pi", 1)[0] + "_pi"
        elif "_t" in nm:
            family = nm.rsplit("_t", 1)[0]
        elif "_b" in nm:
            family = nm.rsplit("_b", 1)[0]
        else:
            family = st["kind"]
        rows.append({"name": nm, "kind": st["kind"], "family": family,
                     "thres": st["thres"], "budget": st.get("budget"),
                     "map50": map50, "map": map5095, "ar": ar5095,
                     "per_class": per_cls,
                     "super_rate": super_rate, "gflops": gflops})
    rows.sort(key=lambda r: r["gflops"])

    # ── pycocotools cross-check / COCO-standard summary ───────────────────────
    if dump:
        coco = dump.summarize(args.coco_out)
        ours = {r["name"]: r for r in rows}
        print("\n=== pycocotools COCOeval vs our matcher ===")
        print(f"{'strategy':<26}{'AP(coco)':>10}{'AP(ours)':>10}{'d':>8}"
              f"{'AR100':>10}{'AR(ours)':>10}{'d':>8}")
        for nm, res in coco.items():
            st_ = res["stats"]; o = ours[nm]
            print(f"{nm:<26}{st_['AP']:>10.4f}{o['map']:>10.4f}{st_['AP']-o['map']:>+8.4f}"
                  f"{st_['AR@100']:>10.4f}{o['ar']:>10.4f}{st_['AR@100']-o['ar']:>+8.4f}")
            o["coco"] = st_
        print(f"[*] COCO dump -> {args.coco_out}")

    hdr = f"{'strategy':<26}{'super%':>8}{'GFLOPs':>9}{'mAP50':>9}{'mAP':>9}"
    lines = [hdr] + [
        f"{r['name']:<26}{r['super_rate']*100:>7.1f}%{r['gflops']:>9.2f}"
        f"{r['map50']:>9.4f}{r['map']:>9.4f}" for r in rows]
    print("\n" + "\n".join(lines))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if fd is not None:
        fd.save(args.frame_dump, {"gflops_base": gb, "gflops_super": gs,
                                  "dataset": args.dataset, "conf": args.conf,
                                  "max_det": args.max_det})

    out.write_text(json.dumps({"gflops_super": gs, "gflops_base": gb,
                               "rows": rows}, indent=2))
    with open(out.with_suffix(".log"), "w") as f:
        f.write(f"# eval_video.py {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# args: {json.dumps(vars(args))}\n")
        f.write(f"# seqs={len(seqs)} frames={total_frames} "
                f"strategies={len(strategies)} base={gb:.2f} super={gs:.2f}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"[*] saved -> {out}\n[*] log -> {out.with_suffix('.log')}")


if __name__ == "__main__":
    main()
