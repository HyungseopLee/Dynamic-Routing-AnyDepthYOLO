"""Streaming depth-routing video evaluation on the Waymo Open Dataset (our
protocol, Waymo labels). Each Waymo segment (~20 s @ 10 Hz FRONT camera) is one
video clip; for each labeled frame we run BOTH the BASE and SUPER detector paths
(shared detection pool), tap the router features, and let each strategy pick a
path causally from the previous frame -- identical protocol to KITTI/BDD, so the
three datasets are directly comparable.

What is dataset-SPECIFIC here (NOT shared with BDD): the segment lister, the Waymo
label parser, the frame source (pre-extracted JPEGs, no .mov decode / rotation /
off-by-one), and the eval class set. What is shared (dataset-AGNOSTIC, imported):
the AP matching/mAP engine (eval_baseline_kitti), the router + PIController, the
feature taps. AP is computed by the SAME engine for BASE and SUPER, so the path
comparison that the router curve rests on is fair.

Waymo 2D classes -> BDD-anydepth detector ids (set by waymo_v2_to_yolo.py):
    vehicle->2 car   pedestrian->0 person   cyclist->1 rider   sign->8 traffic sign
We score exactly this set (WAYMO_EVAL_CLS); the detector's other classes are not
penalised as false positives.

    python -m method_advantage_regress.eval.eval_video_waymo \
        --weight <anydepth.pt> --policies p0=...pt,p1=...pt \
        --waymo_root /media/data/waymo/val --val_cache <cache>.pt \
        --pi --policy_only --no_conf --out outputs/waymo/eval/video_curve.json
"""
import argparse, json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

import eval_baseline_kitti as B
from method_advantage_regress.router.feature_tap import (
    INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS)
# reuse the dataset-agnostic leaves (no protocol logic duplicated)
from method_advantage_regress.eval.eval_video_bdd import (
    PIController, load_policy, grid_vec, pixel_signals)

# Waymo-native taxonomy. Waymo 2D camera boxes label only VEHICLE/PEDESTRIAN/
# CYCLIST (no SIGN), so the finetuned detector has a 3-class head; score all.
WAYMO_NAMES = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}
WAYMO_EVAL_CLS = sorted(WAYMO_NAMES)


def parse_waymo_labels(txt_path):
    """labels/<segment>.txt  (lines: 'frame cls xc yc w h', normalized) ->
    gt_norm[frame] = [(cls, xc, yc, w, h), ...] in NORMALIZED coords. Converted to
    pixel x1y1x2y2 per-frame in the loop once the frame size is known."""
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


def to_pixel_boxes(rows, W, H):
    """[(cls,xc,yc,w,h) norm] -> [(cls,x1,y1,x2,y2) px] for the AP engine."""
    out = []
    for cls, xc, yc, w, h in rows:
        x1 = (xc - w / 2) * W; y1 = (yc - h / 2) * H
        x2 = (xc + w / 2) * W; y2 = (yc + h / 2) * H
        out.append((cls, x1, y1, x2, y2))
    return out


def waymo_frames(seg_img_dir, gt):
    """Yield (frame_idx, bgr) for each labeled frame, in temporal order. Frames are
    pre-extracted JPEGs named <frame:06d>.jpg, so no decode/rotation/alignment
    guesswork (unlike BDD .mov) -- the label index IS the frame file index."""
    for fi in sorted(gt):
        p = seg_img_dir / f"{fi:06d}.jpg"
        if not p.exists():
            continue
        bgr = cv2.imread(str(p))
        if bgr is not None:
            yield fi, bgr


def main():
    B.EVAL_CLS = WAYMO_EVAL_CLS                  # score Waymo's mapped classes only
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waymo")
    ap.add_argument("--weight", required=True)
    ap.add_argument("--policies", default="", help="comma-sep tag=path")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--waymo_root", default="/media/data/waymo/val",
                    help="dir with images/<seg>/<frame>.jpg and labels/<seg>.txt")
    ap.add_argument("--sequences", type=str, nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[1280, 1920])
    ap.add_argument("--grid", default="2")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thresh_clips", type=int, default=25)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--raw_out", default=None)
    ap.add_argument("--policy_only", action="store_true")
    ap.add_argument("--no_conf", action="store_true")
    ap.add_argument("--policy_taus", type=int, default=21)
    ap.add_argument("--val_cache", default=None)
    ap.add_argument("--budgets", default="10,20,30,40,50,60,70,80,90")
    ap.add_argument("--pi", action="store_true")
    ap.add_argument("--pi_targets", default=None)
    ap.add_argument("--pi_kp", type=float, default=2.0)
    ap.add_argument("--pi_ki", type=float, default=0.2)
    ap.add_argument("--pi_beta", type=float, default=0.9)
    ap.add_argument("--pi_tau0", type=float, default=0.0)
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

    # ---- policies (tag -> net) ----
    if args.policies:
        pol_specs = [kv.split("=", 1) for kv in args.policies.split(",")]
    elif args.policy:
        pol_specs = [("policy", args.policy)]
    else:
        pol_specs = []
    nets, feats, gaps = {}, {}, {}
    dummy_in = torch.zeros(2, 768, args.grid[0], args.grid[1], device=device)
    dummy_pr = torch.zeros(2, 640, args.grid[0], args.grid[1], device=device)
    zpid = torch.zeros(2, dtype=torch.long, device=device)
    for tag, path in pol_specs:
        net, ckpt, feat, is_gap = load_policy(path, device)
        net.eval()
        di = dummy_in.mean(dim=(2, 3)) if is_gap else dummy_in
        dp = dummy_pr.mean(dim=(2, 3)) if is_gap else dummy_pr
        with torch.no_grad():
            net(di, None if feat == "input" else dp, zpid)
        net.load_state_dict(ckpt["state_dict"]); net.eval()
        nets[tag] = net; feats[tag] = feat; gaps[tag] = is_gap
    need_pred = any(f != "input" for f in feats.values())
    print(f"[*] policies: {list(nets)} (need_pred={need_pred})")

    captured = {}
    for idx in STATE_LAYERS:
        yolo.model.model[idx].register_forward_hook(
            lambda m, i, o, k=idx: captured.__setitem__(k, o))

    img_root = Path(args.waymo_root) / "images"
    label_dir = Path(args.waymo_root) / "labels"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.txt"))
    if args.limit > 0:
        seqs = seqs[:args.limit]
    eval_seqs = seqs[args.shard_id::args.num_shards] if args.num_shards > 1 else seqs

    # ---- lum/edge percentile thresholds (baselines only) ----
    lum_taus = edge_taus = None
    pcts = list(range(5, 100, 5))
    if not args.policy_only:
        lum_all, edge_all = [], []
        thr_seqs = seqs if args.thresh_clips <= 0 else seqs[:args.thresh_clips]
        for seq in thr_seqs:
            gt = parse_waymo_labels(label_dir / f"{seq}.txt")
            for _, bgr in waymo_frames(img_root / seq, gt):
                l, e = pixel_signals(bgr); lum_all.append(l); edge_all.append(e)
        lum_taus = np.percentile(lum_all, pcts); edge_taus = np.percentile(edge_all, pcts)
        print(f"[*] lum/edge thresholds over {len(lum_all)} frames from {len(thr_seqs)} clips")

    # ---- val-derived policy thresholds (honest budgets) ----
    val_taus = None
    if args.val_cache:
        budgets = [int(b) for b in args.budgets.split(",")]
        vc = torch.load(args.val_cache, map_location="cpu", weights_only=False)
        v_in = vc["input_base"].to(device)
        # pred (neck) features are only present in feat=both caches; backbone-only
        # (feat=input) policies don't need them, so keep None and let the per-policy
        # guard below skip pred. A pred-needing policy on an input-only cache will
        # surface clearly when it tries to use this None.
        v_pr = vc["pred_base"].to(device) if "pred_base" in vc else None
        v_pid = torch.zeros(v_in.shape[0], dtype=torch.long, device=device)
        val_taus = {}
        with torch.no_grad():
            for tag, net in nets.items():
                xi = v_in.mean(dim=(2, 3)) if gaps[tag] else v_in
                pr = None if feats[tag] == "input" else (v_pr.mean(dim=(2, 3)) if gaps[tag] else v_pr)
                ah = net.logit(xi, pr, v_pid).view(-1).cpu().numpy()
                val_taus[tag] = {b: float(np.quantile(ah, 1.0 - b / 100.0)) for b in budgets}
        print(f"[*] val thresholds (budgets={budgets})")

    # ---- strategies ----
    strategies = [dict(name="always_base", kind="const", thres=0, decide=lambda pc, pv, fi: "base"),
                  dict(name="always_super", kind="const", thres=0, decide=lambda pc, pv, fi: "super")]
    if not args.policy_only:
        for p in range(10, 100, 10):
            ps = p / 100.0
            strategies.append(dict(name=f"random_p{p:03d}", kind="random", thres=ps,
                                   decide=lambda pc, pv, fi, ps=ps: "super" if np.random.random() < ps else "base"))
        for kind, taus in (("lum", lum_taus), ("edge", edge_taus)):
            for pc, tau in zip(pcts, taus):
                strategies.append(dict(name=f"{kind}_p{pc:02d}", kind=kind, thres=float(tau),
                                       decide=lambda pc_, pv, fi, tau=tau: "super" if pv < tau else "base"))
        if not args.no_conf:
            conf_taus = [round((i + 1) / 10.0, 2) for i in range(9)]
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
    print(f"[*] shard {args.shard_id}/{args.num_shards}: {len(eval_seqs)} of {len(seqs)} segments, "
          f"{len(strategies)} strategies")

    np.random.seed(0)
    state = {s["name"]: B.StrategyState(s["name"]) for s in strategies}
    one = torch.ones(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, dtype=torch.long, device=device)

    total_frames = 0
    for vi, seq in enumerate(eval_seqs):
        gt_norm = parse_waymo_labels(label_dir / f"{seq}.txt")
        for s in state.values():
            s.prev_choice = None
        for st in strategies:
            if "ctrl" in st:
                st["ctrl"].reset()
        nfr = 0
        for fi_pos, (fidx, bgr) in enumerate(waymo_frames(img_root / seq, gt_norm)):
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
                        "confge10": B.conf_mean_ge(conf, 0.1) if conf.numel() else 0.0}
            sig_super, sig_base = sig(conf_s), sig(conf_b)
            with torch.no_grad():
                av = {}
                for tag, n in nets.items():
                    xs_i, xb_i = (in_s.mean(dim=(2, 3)), in_b.mean(dim=(2, 3))) if gaps[tag] else (in_s, in_b)
                    if feats[tag] == "input":
                        ps_i = pb_i = None
                    else:
                        ps_i, pb_i = (pr_s.mean(dim=(2, 3)), pr_b.mean(dim=(2, 3))) if gaps[tag] else (pr_s, pr_b)
                    av[tag] = {"super": float(n.logit(xs_i, ps_i, one)),
                               "base": float(n.logit(xb_i, pb_i, zero))}

            gts = to_pixel_boxes(gt_norm.get(fidx, []), W, H)
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

    # ---- shard mode ----
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
        f.write(f"# eval_video_waymo.py {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# args: {json.dumps(vars(args))}\n")
        f.write(f"# segments={len(seqs)} frames={total_frames} strategies={len(strategies)}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"[*] saved -> {out}\n[*] log -> {out.with_suffix('.log')}")


if __name__ == "__main__":
    main()
