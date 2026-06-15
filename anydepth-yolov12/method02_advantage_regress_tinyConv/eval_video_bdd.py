"""Video (BDD100K MOT box_track_20 val) AP-vs-FLOPs evaluation for the tiny-conv
depth-routing policy. BDD analogue of eval_video.py (KITTI).

Differences from the KITTI script:
  - frame source : decode .mov with cv2.VideoCapture (no PNG dirs)
  - labels       : one box_track_20 json per video (5fps), parsed to per-frame GT
  - classes      : BDD detector 10-class taxonomy; score only the movable classes
                   MOT annotates (pedestrian,rider,car,bus,truck,bicycle,motorcycle,
                   train) -> drop traffic light/sign so they are not counted as FP
  - cadence      : routing + scoring run at the 5fps labeled frames (the only frames
                   with GT); temporal recursion steps frame->frame at that cadence

Recursive temporal routing: the path used for labeled frame t is decided from
frame t-1's chosen-path signal (confidence / predicted advantage). Every scored
frame runs BOTH BASE and SUPER so all strategies share one detection pool and
differ only in WHICH path they keep + how they route.

Usage:
    python method02_advantage_regress_tinyConv/eval_video_bdd.py \
        --weight finetuned_bdd100k/30e_..._alpha0.2_orig_mAP35.1_33.8.pt \
        --mot_root /media/data/bdd100k_mot/val \
        --policies seed0=outputs/bdd100k/policy_0.pt,seed1=outputs/bdd100k/policy_1.pt \
        --grid 2 --imgsz 720 1280 --conf 0.001 --limit 0
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO

import eval_baseline_kitti as B  # reuse matching / mAP / conf features / energy / state
from method02_advantage_regress_tinyConv.feature_tap import INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS
from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork

# MOT category -> BDD detector class id (matches bdd100k.yaml / det_20 converter).
# Names differ cosmetically (person==pedestrian, bike==bicycle, motor==motorcycle).
MOT_TO_ID = {
    "pedestrian": 0, "rider": 1, "car": 2, "bus": 3, "truck": 4,
    "bicycle": 5, "motorcycle": 6, "train": 9,
}
# Score only classes MOT actually annotates. traffic light(7)/sign(8) are dropped
# so the detector's TL/TS boxes are not penalised as false positives.
BDD_MOT_EVAL_CLS = sorted(set(MOT_TO_ID.values()))


def parse_box_track(json_path):
    """box_track_20 json -> gt_by_frameindex[fi] = [(cls,x1,y1,x2,y2), ...].
    Keyed by the label's 0-based frameIndex (5fps position)."""
    gt = {}
    for fr in json.loads(Path(json_path).read_text()):
        fi = int(fr["frameIndex"])
        boxes = []
        for l in fr.get("labels", []):
            cid = MOT_TO_ID.get(l.get("category"))
            b = l.get("box2d")
            if cid is None or b is None:
                continue
            boxes.append((cid, float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])))
        gt[fi] = boxes
    return gt


def grid_vec(captured, layers, G):
    """Adaptive-pool each tapped map to GxG and concat on channels -> [sumC,G,G]."""
    return torch.cat([F.adaptive_avg_pool2d(captured[i].float(), G).squeeze(0) for i in layers], dim=0)


def pixel_signals(bgr):
    """Cheap pixel stats (no detector): luminance (low->night) and edge density
    (low->sparse). Low signal -> hard scene -> route to SUPER."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:, :, 0]
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return float(y.mean()), float(np.sqrt(gx * gx + gy * gy).mean())


_ANGLE_TO_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def open_upright(mov_path):
    """Open a .mov and return (cap, meta_angle) so that decoded frames are upright
    regardless of the opencv version.

    BDD MOT clips carry a rotation in their container metadata (e.g. 270°). Newer
    opencv auto-applies it (returns landscape 720x1280); older opencv does NOT
    (returns the raw portrait 1280x720). Relying on a fixed shape->ROTATE_90_CW
    rule is therefore version-dependent and silently flips frames upside down on
    the env that already auto-rotates. We instead (1) explicitly request
    auto-orientation and (2) keep the metadata angle as a fallback for builds that
    ignore the flag (see decode_upright)."""
    cap = cv2.VideoCapture(str(mov_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)       # honour rotation metadata
    meta = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    return cap, meta


def decode_upright(frame, meta_angle):
    """Straighten a frame if the build did not auto-apply the rotation metadata."""
    if frame.shape[0] > frame.shape[1] and meta_angle in _ANGLE_TO_ROT:   # still portrait
        return cv2.rotate(frame, _ANGLE_TO_ROT[meta_angle])
    return frame


def decoded_frame_for_label(fi, step):
    """Decoded-frame index for a 5fps box_track label `fi`, the SINGLE source of
    truth for label<->frame alignment (used by both labeled_frames and the
    probe_align.py regression test -- never duplicate this formula).

    box_track_20 labels are 0-based at 5fps; the .mov runs at ~30fps so a label
    step spans `step = fps/5 ~= 6` decoded frames. The naive round(fi*step),
    however, lands ONE label step AHEAD of the actual annotated frame (verified by
    probe_align.py: AP peaks at offset -round(step), recovering mAP50 0.47 -> 0.62).
    We therefore subtract one step. The offset is derived from fps (not hardcoded)
    so clips at other frame rates stay aligned."""
    return max(0, int(round(fi * step)) - int(round(step)))


def labeled_frames(mov_path, gt, fps_label=5.0):
    """Yield (frameIndex, bgr) for each labeled frame, by decoding the .mov and
    sampling the decoded frame aligned to each 5fps label position."""
    cap, meta = open_upright(mov_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = fps / fps_label                       # decoded frames per label step
    targets = {decoded_frame_for_label(fi, step): fi for fi in sorted(gt)}
    di, got = 0, 0
    need = len(targets)
    while got < need:
        ok, frame = cap.read()
        if not ok:
            break
        if di in targets:
            yield targets[di], decode_upright(frame, meta)
            got += 1
        di += 1
    cap.release()


def load_policy(path, device):
    """Load a router checkpoint, auto-selecting the architecture: tinyConv (conv on
    a 2x2 grid; proj weights are 4D) vs GAP-MLP (MLP on a GAP vector; proj weights
    are 2D). Returns (net, ckpt, feat, is_gap). GAP-MLP routers consume the spatial
    mean of the grid features, so callers must pool to [B,C] when is_gap is True."""
    from method01_advantage_regress.policy_net import PolicyNetwork as GapMlpNet
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    sd = ckpt["state_dict"]
    is_gap = not any(k.endswith("weight") and v.dim() == 4 for k, v in sd.items())
    cls = GapMlpNet if is_gap else PolicyNetwork
    net = cls(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
              hidden_dim=a.get("hidden", 64), feat=a.get("feat", "both"),
              norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(device)
    return net, ckpt, a.get("feat", "both"), is_gap


class PIController:
    """Online latency-budget control of the routing threshold tau (Alg. 2 of the
    paper, ported from KITTI to BDD video). The fixed-tau loop serves one operating
    point; this PI layer instead holds a target latency L*(t) and adapts tau in
    closed loop every frame -- no retraining, no val-derived threshold -- so the
    realized latency tracks the budget under a val->video / domain shift (the whole
    BDD problem, where the val-derived oracle tau transfers poorly).

    Latency is normalized to the fraction between the base and super paths,
        ell = 1 if super else 0          (the constant router cost ell_rtr drops out),
    so the target L* is just the desired SUPER-usage fraction in [0,1] and the
    gains kp,ki stay O(0.1-1). Per the paper:
        Lbar <- beta*Lbar + (1-beta)*ell        # EMA of realized latency
        e    <- L* - Lbar ;  I <- I + e
        tau  <- clip(tau0 - kp*e - ki*I)         # lower tau => more super => higher latency
        super if (frame>0 and Ahat > tau)
    Self-contained state, reset per clip; matches decide(prev_choice, prev_value,
    frame_idx)."""

    def __init__(self, target, kp, ki, beta, tau0, tau_lo=-2.0, tau_hi=2.0):
        self.Lstar, self.kp, self.ki, self.beta, self.tau0 = target, kp, ki, beta, tau0
        self.tau_lo, self.tau_hi = tau_lo, tau_hi
        self.reset()

    def reset(self):
        self.I = 0.0
        self.Lbar = 0.5          # EMA init at midpoint (1/2)(ell_base+ell_super)
        self.tau = self.tau0

    def __call__(self, prev_choice, prev_value, frame_idx):
        choice = "super" if (frame_idx > 0 and prev_value > self.tau) else "base"
        ell = 1.0 if choice == "super" else 0.0
        self.Lbar = self.beta * self.Lbar + (1.0 - self.beta) * ell
        e = self.Lstar - self.Lbar
        self.I += e
        self.tau = min(max(self.tau0 - self.kp * e - self.ki * self.I, self.tau_lo), self.tau_hi)
        return choice


def main():
    B.EVAL_CLS = BDD_MOT_EVAL_CLS   # override KITTI taxonomy for matching / mAP

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bdd100k")
    ap.add_argument("--weight", required=True)
    ap.add_argument("--policies", default="", help="comma-sep tag=path (e.g. seed0=...pt,seed1=...pt)")
    ap.add_argument("--policy", default=None, help="single policy (if --policies empty)")
    ap.add_argument("--mot_root", default="/media/data/bdd100k_mot/val")
    ap.add_argument("--sequences", type=str, nargs="*", default=None, help="video stems subset")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--grid", default="2", help="spatial grid 'G' or 'HxW' (match build_cache)")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="cap on number of videos (0=all)")
    ap.add_argument("--thresh_clips", type=int, default=25, help="number of (prefix) clips "
                    "used to estimate lum/edge percentile thresholds; 0 = all. Subset keeps "
                    "thresholds consistent across shards while cutting pre-pass decode.")
    ap.add_argument("--num_shards", type=int, default=1, help="split videos across N tasks "
                    "(round-robin); 1 = single-GPU full run")
    ap.add_argument("--shard_id", type=int, default=0, help="this task's shard index [0,N)")
    ap.add_argument("--raw_out", default=None, help="dump raw per-strategy matches/gt for merge "
                    "(shard mode); when set, final mAP is left to merge_video_shards.py")
    ap.add_argument("--policy_only", action="store_true")
    ap.add_argument("--no_conf", action="store_true")
    ap.add_argument("--policy_taus", type=int, default=21)
    ap.add_argument("--val_cache", default=None, help="derive budget thresholds on val (honest)")
    ap.add_argument("--budgets", default="10,20,30,40,50,60,70,80,90")
    ap.add_argument("--pi", action="store_true", help="add PI latency-budget control "
                    "strategies (Alg. 2): one closed-loop tracker per --pi_targets, no "
                    "val cache needed (online tau adaptation).")
    ap.add_argument("--pi_targets", default=None, help="target SUPER-usage %% setpoints "
                    "for PI control (comma-sep); defaults to --budgets")
    ap.add_argument("--pi_kp", type=float, default=2.0, help="PI proportional gain")
    ap.add_argument("--pi_ki", type=float, default=0.2, help="PI integral gain")
    ap.add_argument("--pi_beta", type=float, default=0.9, help="EMA factor for realized latency")
    ap.add_argument("--pi_tau0", type=float, default=0.0, help="initial routing threshold tau0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.grid = (lambda s: tuple(int(x) for x in s.split("x")) if "x" in s
                 else (int(s), int(s)))(str(args.grid))
    _base = Path(__file__).resolve().parent / "outputs" / args.dataset
    if args.out is None:
        args.out = str(_base / "eval" / "video_curve.json")

    device = args.device if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.weight, task="detect")
    yolo.model.to(device).eval()
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super, skip_base = [False] * N, [True] * N
    gs, gb = B.measure_flops_super_base(yolo.model, args.imgsz, device)
    print(f"[*] GFLOPs super={gs:.2f} base={gb:.2f}  EVAL_CLS={B.EVAL_CLS}")

    # policies (tag -> net). all feat=input here, so pred features are unused.
    if args.policies:
        pol_specs = [kv.split("=", 1) for kv in args.policies.split(",")]
    elif args.policy:
        pol_specs = [("policy", args.policy)]
    else:
        pol_specs = []
    nets, feats, gaps = {}, {}, {}
    in_c = 768
    dummy_in = torch.zeros(2, in_c, args.grid[0], args.grid[1], device=device)
    dummy_pr = torch.zeros(2, 640, args.grid[0], args.grid[1], device=device)
    zpid = torch.zeros(2, dtype=torch.long, device=device)
    for tag, path in pol_specs:
        net, ckpt, feat, is_gap = load_policy(path, device)
        net.eval()
        di = dummy_in.mean(dim=(2, 3)) if is_gap else dummy_in   # GAP-MLP wants [B,C]
        dp = dummy_pr.mean(dim=(2, 3)) if is_gap else dummy_pr
        with torch.no_grad():
            net(di, None if feat == "input" else dp, zpid)        # materialise lazy shapes
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        nets[tag] = net; feats[tag] = feat; gaps[tag] = is_gap
    need_pred = any(f != "input" for f in feats.values())  # tap PRED grid only when a policy needs it
    print(f"[*] policies: {list(nets)} (need_pred={need_pred})")

    captured = {}
    for idx in STATE_LAYERS:
        yolo.model.model[idx].register_forward_hook(
            lambda m, i, o, k=idx: captured.__setitem__(k, o))

    label_dir = Path(args.mot_root) / "labels"
    video_dir = Path(args.mot_root) / "videos"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.json"))
    if args.limit > 0:
        seqs = seqs[:args.limit]
    # videos this task actually evaluates (round-robin shard); the lum/edge
    # threshold pre-pass below still runs over ALL videos so the percentile
    # thresholds are identical across shards (mergeable).
    eval_seqs = seqs[args.shard_id::args.num_shards] if args.num_shards > 1 else seqs

    # pre-pass for luminance/edge percentile thresholds (over labeled frames)
    lum_taus = edge_taus = None
    pcts = list(range(5, 100, 5))
    if not args.policy_only:
        lum_all, edge_all = [], []
        # lum/edge percentile thresholds are scene statistics that are stable from a
        # subset, so we estimate them from the first `thresh_clips` videos instead of
        # decoding all of them. The subset is the deterministic global prefix seqs[:N]
        # (identical in every shard) so the lum/edge baseline taus stay consistent and
        # mergeable across shards -- while cutting the pre-pass decode by ~num_clips/N.
        thr_seqs = seqs if args.thresh_clips <= 0 else seqs[:args.thresh_clips]
        for seq in thr_seqs:
            gt = parse_box_track(label_dir / f"{seq}.json")
            for _, bgr in labeled_frames(video_dir / f"{seq}.mov", gt):
                l, e = pixel_signals(bgr); lum_all.append(l); edge_all.append(e)
        lum_taus = np.percentile(lum_all, pcts); edge_taus = np.percentile(edge_all, pcts)
        print(f"[*] luminance [{min(lum_all):.1f},{max(lum_all):.1f}] "
              f"edge [{min(edge_all):.1f},{max(edge_all):.1f}] over {len(lum_all)} frames "
              f"from {len(thr_seqs)} clips")

    # val-derived policy thresholds (honest: tau fixed on val images, applied on video)
    val_taus = None
    if args.val_cache:
        budgets = [int(b) for b in args.budgets.split(",")]
        vc = torch.load(args.val_cache, map_location="cpu", weights_only=False)
        v_in = vc["input_base"].to(device); v_pr = vc["pred_base"].to(device)
        v_pid = torch.zeros(v_in.shape[0], dtype=torch.long, device=device)
        val_taus = {}
        with torch.no_grad():
            for tag, net in nets.items():
                xi = v_in.mean(dim=(2, 3)) if gaps[tag] else v_in     # GAP-MLP wants [B,C]
                pr = None if feats[tag] == "input" else (v_pr.mean(dim=(2, 3)) if gaps[tag] else v_pr)
                ah = net.logit(xi, pr, v_pid).view(-1).cpu().numpy()
                val_taus[tag] = {b: float(np.quantile(ah, 1.0 - b / 100.0)) for b in budgets}
        print(f"[*] val thresholds (budgets={budgets})")

    # ---- strategies ----
    strategies = [dict(name="always_base", kind="const", thres=0, decide=lambda pc, pv, fi: "base"),
                  dict(name="always_super", kind="const", thres=0, decide=lambda pc, pv, fi: "super")]
    if not args.policy_only:
        for p in range(0, 101, 10):
            ps = p / 100.0
            strategies.append(dict(name=f"random_p{p:03d}", kind="random", thres=ps,
                                   decide=lambda pc, pv, fi, ps=ps: "super" if np.random.random() < ps else "base"))
        for kind, taus in (("lum", lum_taus), ("edge", edge_taus)):
            for pc, tau in zip(pcts, taus):
                strategies.append(dict(name=f"{kind}_p{pc:02d}", kind=kind, thres=float(tau),
                                       decide=lambda pc_, pv, fi, tau=tau: "super" if pv < tau else "base"))
        if not args.no_conf:
            # conftop20 only (drop confge10) on a coarser tau grid: the confidence
            # baseline is just a reference curve, so 9 thresholds trace it fine while
            # cutting the per-frame matches_multi accumulation that drives the OOM at
            # conf=0.001 (38 conf strategies -> 9).
            conf_taus = [round((i + 1) / 10.0, 2) for i in range(9)]  # 0.1..0.9
            for kind in ("conftop20",):
                for tau in conf_taus:
                    strategies.append(dict(name=f"{kind}_t{int(tau*100):02d}", kind=kind, thres=tau,
                                           decide=lambda pc, pv, fi, tau=tau: "base" if (fi == 0 or pv >= tau) else "super"))
    policy_taus = [round(-0.4 + (2.0 / (args.policy_taus - 1)) * i, 3) for i in range(args.policy_taus)]
    for tag in nets:
        pname = f"policy_{tag}"
        if val_taus is not None:
            for b, tau in val_taus[tag].items():
                strategies.append(dict(name=f"{pname}_b{b:02d}", kind="policy", ptag=tag,
                                       thres=tau, budget=b,
                                       decide=lambda pc, pv, fi, tau=tau: "super" if (fi > 0 and pv > tau) else "base"))
        else:
            for tau in policy_taus:
                strategies.append(dict(name=f"{pname}_t{int(tau*100):+04d}", kind="policy", ptag=tag,
                                       thres=tau,
                                       decide=lambda pc, pv, fi, tau=tau: "super" if (fi > 0 and pv > tau) else "base"))
        if args.pi:
            pi_targets = [int(t) for t in (args.pi_targets or args.budgets).split(",")]
            for t in pi_targets:
                ctrl = PIController(target=t / 100.0, kp=args.pi_kp, ki=args.pi_ki,
                                    beta=args.pi_beta, tau0=args.pi_tau0)
                strategies.append(dict(name=f"{pname}_pi{t:02d}", kind="policy", ptag=tag,
                                       thres=args.pi_tau0, budget=t, ctrl=ctrl, decide=ctrl))
    print(f"[*] shard {args.shard_id}/{args.num_shards}: {len(eval_seqs)} of {len(seqs)} videos, "
          f"{len(strategies)} strategies")

    np.random.seed(0)
    state = {s["name"]: B.StrategyState(s["name"]) for s in strategies}
    one = torch.ones(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, dtype=torch.long, device=device)

    total_frames = 0
    for vi, seq in enumerate(eval_seqs):
        gt = parse_box_track(label_dir / f"{seq}.json")
        for s in state.values():
            s.prev_choice = None
        for st in strategies:                 # reset PI controller integral/EMA per clip
            if "ctrl" in st:
                st["ctrl"].reset()
        nfr = 0
        for fi_pos, (fidx, bgr) in enumerate(labeled_frames(video_dir / f"{seq}.mov", gt)):
            total_frames += 1; nfr += 1
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
                        "confge10": B.conf_mean_ge(conf, 0.1) if conf.numel() else 0.0}
            sig_super, sig_base = sig(conf_s), sig(conf_b)
            with torch.no_grad():
                av = {}
                for tag, n in nets.items():
                    # GAP-MLP routers consume the spatial mean of the grid features
                    xs_i, xb_i = (in_s.mean(dim=(2, 3)), in_b.mean(dim=(2, 3))) if gaps[tag] else (in_s, in_b)
                    if feats[tag] == "input":
                        ps_i = pb_i = None
                    else:
                        ps_i, pb_i = (pr_s.mean(dim=(2, 3)), pr_b.mean(dim=(2, 3))) if gaps[tag] else (pr_s, pr_b)
                    av[tag] = {"super": float(n.logit(xs_i, ps_i, one)),
                               "base": float(n.logit(xb_i, pb_i, zero))}

            gts = gt.get(fidx, [])
            for st in strategies:
                kind, decide = st["kind"], st["decide"]
                s = state[st["name"]]
                if kind in ("conftop20", "confge10"):
                    pv = (sig_super[kind] if s.prev_choice == "super" else sig_base[kind]) if s.prev_choice else 0.0
                elif kind == "policy":
                    avt = av[st["ptag"]]
                    pv = (avt["super"] if s.prev_choice == "super" else avt["base"]) if s.prev_choice else 0.0
                elif kind == "lum":
                    pv = lum_cur
                elif kind == "edge":
                    pv = edge_cur
                else:
                    pv = 0.0
                choice = decide(s.prev_choice, pv, fi_pos)
                s.prev_choice = choice
                s.n_super += (choice == "super"); s.n_base += (choice == "base")
                m, gtc = B.match_frame_multi_iou(preds[choice], gts)
                s.matches_multi.extend(m)
                for cls_id, cnt in gtc.items():
                    s.gt_count[cls_id] += cnt
        print(f"[{vi+1}/{len(eval_seqs)} {seq}] {nfr} labeled frames")

    # ---- shard mode: dump raw per-strategy matches/gt and stop (merge later) ----
    if args.raw_out:
        raw = {"gflops_super": gs, "gflops_base": gb, "total_frames": total_frames,
               "meta": {s["name"]: {"kind": s["kind"], "thres": s["thres"],
                                    "budget": s.get("budget")} for s in strategies},
               "state": {s["name"]: {"matches_multi": state[s["name"]].matches_multi,
                                     "gt_count": dict(state[s["name"]].gt_count),
                                     "n_super": state[s["name"]].n_super,
                                     "n_base": state[s["name"]].n_base}
                         for s in strategies}}
        rp = Path(args.raw_out); rp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw, rp)
        print(f"[*] shard {args.shard_id} raw -> {rp}")
        return

    # ---- aggregate ----
    rows = []
    for st in strategies:
        s = state[st["name"]]
        _, map50, map5095 = B.dataset_map_multi_iou(s.matches_multi, s.gt_count)
        n = s.n_super + s.n_base
        super_rate = s.n_super / max(n, 1)
        gflops = super_rate * gs + (1 - super_rate) * gb
        if "_pi" in st["name"]:
            family = st["name"].rsplit("_pi", 1)[0] + "_pi"
        elif "_t" in st["name"]:
            family = st["name"].rsplit("_t", 1)[0]
        elif "_b" in st["name"]:
            family = st["name"].rsplit("_b", 1)[0]
        else:
            family = st["kind"]
        rows.append({"name": st["name"], "kind": st["kind"], "family": family,
                     "thres": st["thres"], "budget": st.get("budget"),
                     "map50": map50, "map": map5095,
                     "super_rate": super_rate, "gflops": gflops})
    rows.sort(key=lambda r: r["gflops"])

    hdr = f"{'strategy':<24}{'super%':>8}{'GFLOPs':>9}{'mAP50':>9}{'mAP':>9}"
    lines = [hdr] + [f"{r['name']:<24}{r['super_rate']*100:>7.1f}%{r['gflops']:>9.2f}"
                     f"{r['map50']:>9.4f}{r['map']:>9.4f}" for r in rows]
    print("\n" + "\n".join(lines))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gflops_super": gs, "gflops_base": gb, "rows": rows}, indent=2))
    with open(out.with_suffix(".log"), "w") as f:
        f.write(f"# eval_video_bdd.py {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# args: {json.dumps(vars(args))}\n")
        f.write(f"# videos={len(seqs)} frames={total_frames} strategies={len(strategies)} "
                f"gflops_base={gb:.2f} gflops_super={gs:.2f}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"[*] saved -> {out}\n[*] log -> {out.with_suffix('.log')}")


if __name__ == "__main__":
    main()
