"""Video (KITTI-tracking) AP-vs-FLOPs evaluation for method01 (path B, regress).

Recursive temporal routing: the path used for frame t is decided from frame t-1's
prediction (the previous frame's chosen-path feature / confidence). Every frame
runs BOTH Base and Super forwards so all strategies score on the same pool;
each strategy only differs in WHICH path's detections it keeps and how it routes.

Strategies compared:
  - always-BASE / always-SUPER            (curve endpoints)
  - random_pXXX  (P(super)=0,10,...,100%)  -> random baseline curve
  - conf_tXX     (prev-frame mean confidence threshold) -> confidence baseline
  - policy_tXX   (our regress policy: predicted advantage A-hat thresholded)
  - ORACLE       (per-frame argmin detection... not available w/o GT loss here;
                  instead we report the achievable front via random/best curves)

Reuses label parsing / matching / mAP from eval_baseline_kitti.py.

Usage:
    python method02_advantage_regress_tinyConv/eval_video.py \
        --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
        --policy runs/kitti/policy/policy_regress.pt \
        --kitti_root /media/data/kitti-tracking \
        --imgsz 384 1248 --conf 0.001 \
        --out runs/kitti/policy-eval/video_curve.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from ultralytics import YOLO

import eval_baseline_kitti as B  # reuse label parsing / matching / mAP / conf features
from method02_advantage_regress_tinyConv.feature_tap import INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS
from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    net = PolicyNetwork(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                        hidden_dim=a.get("hidden", 128), feat=a.get("feat", "both"),
                        norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.0)).to(device)
    return net, ckpt


import torch.nn.functional as F


def grid_vec(captured, layers, G):
    """Adaptive-pool each tapped map to GxG and concat on channels -> [sumC, G, G]
    (matches the spatial cache the tiny-conv policy was trained on)."""
    return torch.cat([F.adaptive_avg_pool2d(captured[i].float(), G).squeeze(0) for i in layers], dim=0)


def pixel_signals(bgr):
    """WACV pixel-statistics signals (computed on the raw frame, CPU, no detector):
       luminance = mean of YUV luma channel       (low -> night)
       edge_density = mean Sobel gradient magnitude (low -> sparse/low-contrast)
    Low signal -> hard/sparse scene -> route to SUPER (Eq.7-9)."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:, :, 0]
    lum = float(y.mean())
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.sqrt(gx * gx + gy * gy).mean())
    return lum, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti", help="output scope: outputs/<dataset>/")
    ap.add_argument("--weight", default="runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--policies", default="", help="comma-sep tag=path list to compare "
                    "multiple policies in one pass, e.g. input=...pt,pred=...pt,both=...pt")
    ap.add_argument("--kitti_root", default="/media/data/kitti-tracking")
    ap.add_argument("--sequences", type=str, nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1248])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--policy_only", action="store_true",
                    help="only policy strategies + endpoints (skip conf/lum/edge/random) -- "
                         "use for multi-policy ablation to cut strategy count / memory")
    ap.add_argument("--no_conf", action="store_true",
                    help="skip confidence-based routing baselines (conftop20/confge10) only; "
                         "keep random/lum/edge. Lets --conf be set high (few boxes -> light) "
                         "since conf-routing is the baseline that needs a low-conf box distribution")
    ap.add_argument("--only_conf", action="store_true",
                    help="evaluate ONLY the confidence-routing baselines (+ endpoints); skip "
                         "policy/random/lum/edge. Use to run the heavy conf baseline separately "
                         "(possibly at its own --conf) and merge into the main figure at plot time")
    ap.add_argument("--grid", default="4",
                    help="spatial grid for the tiny-conv policy input: 'G' (square) or 'HxW' "
                         "(rectangular, e.g. '2x6'). Must match build_cache --grid.")
    ap.add_argument("--keep_layers", default="", help="comma-sep subset of backbone tapped "
                    "layers (e.g. '4'); empty = all. Must match the policy's training --keep_layers.")
    ap.add_argument("--policy_taus", type=int, default=21,
                    help="number of policy threshold points (-0.4..+1.6)")
    ap.add_argument("--val_taus_json", default=None,
                    help="thresholds precomputed on val by get_thresholds.py "
                         "(honest pipeline: tau fixed on val, only applied on video). Preferred.")
    ap.add_argument("--val_cache", default=None,
                    help="fallback: derive thresholds inline from this val cache "
                         "(quantile of A-hat per target budget). Prefer --val_taus_json.")
    ap.add_argument("--budgets", default="10,20,30,40,50,60,70,80,90",
                    help="comma-sep target SUPER-rate %% for --val_cache mode")
    ap.add_argument("--conf_taus", type=int, default=19,
                    help="number of confidence-routing threshold points per variant "
                         "(spread over (0,1); lower = lighter / fewer strategies)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # grid: 'G' (square) or 'HxW' (rectangular) -> (Gh, Gw) tuple for adaptive pool
    args.grid = (lambda s: tuple(int(x) for x in s.split("x")) if "x" in s
                 else (int(s), int(s)))(str(args.grid))
    _base = Path(__file__).resolve().parent / "outputs" / args.dataset
    if args.policy is None: args.policy = str(_base / "policy_regress.pt")
    if args.out is None:    args.out = str(_base / "eval" / "video_curve.json")

    device = args.device if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.weight, task="detect")
    yolo.model.to(device).eval()
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip_super, skip_base = [False] * N, [True] * N
    gs, gb = B.measure_flops_super_base(yolo.model, args.imgsz, device)
    print(f"[*] GFLOPs super={gs:.2f} base={gb:.2f}")

    # one or many policies (tag -> net); all share the per-frame GAP features.
    if args.policies:
        pol_specs = [kv.split("=", 1) for kv in args.policies.split(",")]
    else:
        pol_specs = [("policy", args.policy)]
    nets = {}
    keep = [int(x) for x in args.keep_layers.split(",")] if args.keep_layers else []
    input_layers = [L for L in INPUT_LEVEL_LAYERS if (not keep or L in keep)]
    in_c = (768 // len(INPUT_LEVEL_LAYERS)) * len(input_layers)
    dummy_in = torch.zeros(2, in_c, args.grid[0], args.grid[1], device=device)
    dummy_pr = torch.zeros(2, 640, args.grid[0], args.grid[1], device=device)
    for tag, path in pol_specs:
        net, _ = load_policy(path, device)
        net.eval()  # BN eval mode -> batch size 1 ok
        with torch.no_grad():
            net(dummy_in, dummy_pr, torch.zeros(2, dtype=torch.long, device=device))
        net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
        net.eval()
        nets[tag] = net
    print(f"[*] policies: {list(nets)}")

    # hooks to capture GAP features of STATE_LAYERS
    captured = {}
    for idx in STATE_LAYERS:
        yolo.model.model[idx].register_forward_hook(
            lambda m, i, o, k=idx: captured.__setitem__(k, o))

    label_dir = Path(args.kitti_root) / "training" / "label_02"
    seqs = args.sequences or sorted(f.stem for f in label_dir.glob("*.txt"))

    def frames_of(seq):
        d = Path(args.kitti_root) / "training" / "image_02" / seq
        fr = sorted(d.glob("*.png")) or sorted(d.glob("*.jpg"))
        return fr[:args.limit] if args.limit > 0 else fr

    # pre-pass: collect luminance / edge density over all frames to set percentile
    # thresholds (WACV sweeps percentiles of each pixel-statistics signal).
    lum_all, edge_all = [], []
    if not args.policy_only and not args.only_conf:
        for seq in seqs:
            for fp in frames_of(seq):
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    continue
                l, e = pixel_signals(bgr)
                lum_all.append(l); edge_all.append(e)
        pcts = list(range(5, 100, 5))
        lum_taus = np.percentile(lum_all, pcts)
        edge_taus = np.percentile(edge_all, pcts)
        print(f"[*] luminance range [{min(lum_all):.1f},{max(lum_all):.1f}], "
              f"edge range [{min(edge_all):.1f},{max(edge_all):.1f}]")

    # strategies: dict(name, kind, thres, decide). decide(pc, pv, fi).
    #   conftop20/confge10 -> prev-frame chosen-path confidence (recursive)
    #   policy             -> prev-frame predicted advantage A-hat (recursive)
    #   lum/edge           -> CURRENT-frame pixel signal (causal, memoryless): super if < tau
    strategies = [dict(name="always_base", kind="const", thres=0, decide=lambda pc, pv, fi: "base"),
                  dict(name="always_super", kind="const", thres=0, decide=lambda pc, pv, fi: "super")]
    if not args.policy_only and not args.only_conf:
        for p in range(0, 101, 10):
            ps = p / 100.0
            strategies.append(dict(name=f"random_p{p:03d}", kind="random", thres=ps,
                                   decide=lambda pc, pv, fi, ps=ps: "super" if np.random.random() < ps else "base"))
        for kind, taus in (("lum", lum_taus), ("edge", edge_taus)):
            for pc, tau in zip(pcts, taus):
                strategies.append(dict(name=f"{kind}_p{pc:02d}", kind=kind, thres=float(tau),
                                       decide=lambda pc_, pv, fi, tau=tau: "super" if pv < tau else "base"))
    # confidence-routing: included unless --no_conf; the ONLY families when --only_conf
    if (not args.no_conf and not args.policy_only) or args.only_conf:
        conf_taus = [round((i + 1) / (args.conf_taus + 1), 3) for i in range(args.conf_taus)]
        for kind in ("conftop20", "confge10"):
            for tau in conf_taus:
                strategies.append(dict(name=f"{kind}_t{int(tau*100):02d}", kind=kind, thres=tau,
                                       decide=lambda pc, pv, fi, tau=tau: "base" if (fi == 0 or pv >= tau) else "super"))
    # per-policy thresholds: either a global sweep (ablation) OR val-derived budgets
    # (honest pipeline -- tau fixed on val, only applied here on video).
    val_taus = None
    if args.val_taus_json:
        td = json.loads(Path(args.val_taus_json).read_text())["thresholds"]
        val_taus = {ptag: {int(b): float(t) for b, t in bt.items()} for ptag, bt in td.items()}
        print(f"[*] loaded val thresholds from {Path(args.val_taus_json).name}")
    elif args.val_cache:
        budgets = [int(b) for b in args.budgets.split(",")]
        vc = torch.load(args.val_cache, map_location="cpu", weights_only=False)
        v_in, v_pr = vc["input_base"].to(device), vc["pred_base"].to(device)
        v_pid = torch.zeros(v_in.shape[0], dtype=torch.long, device=device)
        val_taus = {}
        with torch.no_grad():
            for ptag, net in nets.items():
                ah = net.logit(v_in, v_pr, v_pid).view(-1).cpu().numpy()
                # tau_r = quantile(A-hat, 1-r): fraction r of val frames have A-hat>tau -> SUPER
                val_taus[ptag] = {b: float(np.quantile(ah, 1.0 - b / 100.0)) for b in budgets}
        print(f"[*] val-derived thresholds (budgets={budgets}) from {Path(args.val_cache).name}")

    policy_taus = [round(-0.4 + (2.0 / (args.policy_taus - 1)) * i, 3) for i in range(args.policy_taus)]
    for ptag in (nets if not args.only_conf else {}):
        pname = "policy" if ptag == "policy" else f"policy_{ptag}"
        if val_taus is not None:
            for b, tau in val_taus[ptag].items():
                strategies.append(dict(name=f"{pname}_b{b:02d}", kind="policy",
                                       ptag=ptag, thres=tau, budget=b,
                                       decide=lambda pc, pv, fi, tau=tau: "super" if (fi > 0 and pv > tau) else "base"))
        else:
            for tau in policy_taus:  # -0.4 .. +1.6
                strategies.append(dict(name=f"{pname}_t{int(tau*100):+04d}", kind="policy",
                                       ptag=ptag, thres=tau,
                                       decide=lambda pc, pv, fi, tau=tau: "super" if (fi > 0 and pv > tau) else "base"))

    print(f"[*] {len(seqs)} sequences, {len(strategies)} strategies")

    np.random.seed(0)
    state = {s["name"]: B.StrategyState(s["name"]) for s in strategies}
    energy_monitor = B.EnergyMonitor()

    # GPU warmup so the first real frame's latency isn't a cold-start outlier
    warm = np.zeros((args.imgsz[0], args.imgsz[1], 3), dtype=np.uint8)
    for sk in (skip_super, skip_base):
        for _ in range(3):
            yolo.predict(source=warm, imgsz=tuple(args.imgsz), conf=args.conf,
                         iou=0.7, skip=sk, verbose=False, device=device)

    total_frames = 0
    for seq in seqs:
        img_dir = Path(args.kitti_root) / "training" / "image_02" / seq
        lpath = label_dir / f"{seq}.txt"
        if not img_dir.exists() or not lpath.exists():
            print(f"[!] skip {seq}"); continue
        gt_by_frame, dc_by_frame = B.parse_kitti_labels(lpath)
        frames = frames_of(seq)
        for s in state.values():
            s.prev_choice = None; s.prev_val_super = None; s.prev_val_base = None
        print(f"[seq {seq}] {len(frames)} frames")

        for fi, fpath in enumerate(frames):
            frame_idx = int(fpath.stem)
            bgr = cv2.imread(str(fpath))
            if bgr is None:
                continue
            total_frames += 1
            lum_cur, edge_cur = pixel_signals(bgr)  # current-frame pixel stats (causal)

            captured.clear()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            p0 = energy_monitor.power_mw(); t0 = time.perf_counter()
            r_super = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                                   iou=0.7, skip=skip_super, verbose=False, device=device)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter(); p1 = energy_monitor.power_mw()
            in_s = grid_vec(captured, input_layers, args.grid).unsqueeze(0)
            pr_s = grid_vec(captured, PRED_LEVEL_LAYERS, args.grid).unsqueeze(0)
            captured.clear()
            r_base = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf,
                                  iou=0.7, skip=skip_base, verbose=False, device=device)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter(); p2 = energy_monitor.power_mw()
            in_b = grid_vec(captured, input_layers, args.grid).unsqueeze(0)
            pr_b = grid_vec(captured, PRED_LEVEL_LAYERS, args.grid).unsqueeze(0)
            lat = {"super": (t1 - t0) * 1000.0, "base": (t2 - t1) * 1000.0}
            ener = {"super": 0.5 * (p0 + p1) * (t1 - t0),   # mW * s = mJ
                    "base": 0.5 * (p1 + p2) * (t2 - t1)}

            preds = {"super": B.boxes_to_preds(r_super), "base": B.boxes_to_preds(r_base)}
            conf_s = r_super.boxes.conf.cpu() if (r_super.boxes is not None and len(r_super.boxes)) else torch.empty(0)
            conf_b = r_base.boxes.conf.cpu() if (r_base.boxes is not None and len(r_base.boxes)) else torch.empty(0)
            # two confidence signals per path (conf=0.001 NMS keeps many low-conf
            # boxes, so all-box mean is uninformative):
            #   top20  = mean of top-20 confidences
            #   ge10   = mean of confidences >= 0.1
            def sig(conf):
                top20 = B.conf_top_k_mean(conf, 20) if conf.numel() else 0.0
                ge10 = B.conf_mean_ge(conf, 0.1) if conf.numel() else 0.0
                return {"conftop20": top20, "confge10": ge10}
            sig_super, sig_base = sig(conf_s), sig(conf_b)
            one = torch.ones(1, dtype=torch.long, device=device)
            zero = torch.zeros(1, dtype=torch.long, device=device)
            # time the router forward(s) so its overhead is charged to the policy's
            # latency/energy. Deploy runs ONE router forward per frame (on the chosen
            # path), so per-frame overhead = total / (2 logits * n_nets).
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pr0 = energy_monitor.power_mw(); tr0 = time.perf_counter()
            with torch.no_grad():
                av = {tag: {"super": float(n.logit(in_s, pr_s, one).view(-1)),
                            "base": float(n.logit(in_b, pr_b, zero).view(-1))}
                      for tag, n in nets.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tr1 = time.perf_counter(); pr1 = energy_monitor.power_mw()
            _nf = max(1, 2 * len(nets))
            router_lat_ms = (tr1 - tr0) * 1000.0 / _nf
            router_ener_mj = 0.5 * (pr0 + pr1) * (tr1 - tr0) / _nf

            gts = gt_by_frame.get(frame_idx, [])
            dc = dc_by_frame.get(frame_idx, [])
            for st in strategies:
                name, kind, decide = st["name"], st["kind"], st["decide"]
                s = state[name]
                # value carried from previous frame's CHOSEN path, by kind
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
                choice = decide(s.prev_choice, pv, fi)
                s.prev_choice = choice
                s.n_super += (choice == "super"); s.n_base += (choice == "base")
                # charge the router's own forward cost to policy strategies only
                # (lum/edge/conf/random use cheap pixel/conf stats, no learned net)
                rl = router_lat_ms if kind == "policy" else 0.0
                re = router_ener_mj if kind == "policy" else 0.0
                s.latency_ms.append(lat[choice] + rl); s.energy_mj.append(ener[choice] + re)

                p = B.filter_dontcare(preds[choice], dc) if dc else preds[choice]
                m, gtc = B.match_frame_multi_iou(p, gts)
                s.matches_multi.extend(m)
                for cls_id, cnt in gtc.items():
                    s.gt_count[cls_id] += cnt

    # aggregate
    rows = []
    for st in strategies:
        name = st["name"]
        s = state[name]
        _, map50, map5095 = B.dataset_map_multi_iou(s.matches_multi, s.gt_count)
        n = s.n_super + s.n_base
        super_rate = s.n_super / max(n, 1)
        gflops = super_rate * gs + (1 - super_rate) * gb
        mean_lat = float(np.mean(s.latency_ms)) if s.latency_ms else 0.0
        mean_e = float(np.mean(s.energy_mj)) if s.energy_mj else 0.0
        # family = curve grouping (policy_input/pred/both stay distinct)
        if "_t" in name:
            family = name.rsplit("_t", 1)[0]
        elif "_b" in name:
            family = name.rsplit("_b", 1)[0]
        else:
            family = st["kind"]
        rows.append({"name": name, "kind": st["kind"], "family": family, "thres": st["thres"],
                     "budget": st.get("budget"),
                     "map50": map50, "map": map5095,
                     "super_rate": super_rate, "gflops": gflops,
                     "latency_ms": mean_lat, "fps": (1000.0 / mean_lat if mean_lat else 0.0),
                     "energy_mj": mean_e})

    rows.sort(key=lambda r: r["gflops"])
    # build the per-strategy table as text, then both print AND log it (persistent
    # evidence: a .log next to the output json with the full config + results).
    hdr = f"{'strategy':<22}{'super%':>8}{'GFLOPs':>9}{'mAP50':>9}{'mAP':>9}{'ms':>8}{'FPS':>7}{'mJ':>9}"
    lines = [hdr] + [
        f"{r['name']:<22}{r['super_rate']*100:>7.1f}%{r['gflops']:>9.2f}{r['map50']:>9.4f}"
        f"{r['map']:>9.4f}{r['latency_ms']:>8.2f}{r['fps']:>7.1f}{r['energy_mj']:>9.1f}"
        for r in rows]
    print("\n" + "\n".join(lines))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gflops_super": gs, "gflops_base": gb, "rows": rows}, indent=2))
    print(f"[*] saved -> {out}")

    logpath = out.with_suffix(".log")
    with open(logpath, "w") as f:
        f.write(f"# eval_video.py run {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# args: {json.dumps(vars(args))}\n")
        f.write(f"# sequences={len(seqs)} total_frames={total_frames} "
                f"strategies={len(strategies)} gflops_base={gb:.2f} gflops_super={gs:.2f}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"[*] log  -> {logpath}")

    # table: strategies matching or exceeding always-super accuracy (mAP50)
    sup = next((r for r in rows if r["name"] == "always_super"), None)
    if sup:
        qual = [r for r in rows if r["name"] not in ("always_super", "always_base")
                and r["map50"] >= sup["map50"] - 1e-9]
        qual.sort(key=lambda r: r["gflops"])
        md = ["# Strategies matching/exceeding always-SUPER mAP50",
              f"always_super: mAP50={sup['map50']:.4f} mAP={sup['map']:.4f} "
              f"GFLOPs={sup['gflops']:.2f} {sup['latency_ms']:.2f}ms "
              f"{sup['fps']:.1f}FPS {sup['energy_mj']:.1f}mJ\n",
              "| strategy | threshold | super% | GFLOPs | mAP50 | mAP | latency(ms) | FPS | energy(mJ) | ΔFLOPs% |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in qual:
            dfl = (r["gflops"] - sup["gflops"]) / sup["gflops"] * 100
            md.append(f"| {r['name']} | {r['thres']:g} | {r['super_rate']*100:.1f} | "
                      f"{r['gflops']:.2f} | {r['map50']:.4f} | {r['map']:.4f} | "
                      f"{r['latency_ms']:.2f} | {r['fps']:.1f} | {r['energy_mj']:.1f} | {dfl:+.1f} |")
        tbl_path = out.with_name(out.stem + "_super_table.md")
        tbl_path.write_text("\n".join(md))
        print(f"\n[*] {len(qual)} strategies >= always-super mAP50 -> {tbl_path}")
        print("\n".join(md))

        # concise auto-summary: best (lowest-GFLOPs) policy point that matches/exceeds
        # always-SUPER, reported per metric (mAP50 and mAP).
        bas = next((r for r in rows if r["name"] == "always_base"), None)
        sm = ["# Eval summary",
              f"run: {datetime.now().isoformat(timespec='seconds')}  |  "
              f"sequences={len(seqs)} frames={total_frames} strategies={len(strategies)} "
              f"conf={args.conf}",
              f"detector: BASE {gb:.2f} / SUPER {gs:.2f} GFLOPs\n",
              "| reference | mAP50 | mAP | GFLOPs | latency(ms) | FPS | energy(mJ) |",
              "|---|---|---|---|---|---|---|",
              f"| always-SUPER | {sup['map50']:.4f} | {sup['map']:.4f} | {sup['gflops']:.2f} | "
              f"{sup['latency_ms']:.2f} | {sup['fps']:.1f} | {sup['energy_mj']:.1f} |"]
        if bas:
            sm.append(f"| always-BASE | {bas['map50']:.4f} | {bas['map']:.4f} | {bas['gflops']:.2f} | "
                      f"{bas['latency_ms']:.2f} | {bas['fps']:.1f} | {bas['energy_mj']:.1f} |")
        sm.append("\nBest policy point matching/exceeding always-SUPER (lowest GFLOPs):")
        sm.append("| metric | strategy | super% | GFLOPs | value | Δvalue | FPS | ΔFLOPs% | ΔFPS% | Δenergy% |")
        sm.append("|---|---|---|---|---|---|---|---|---|---|")
        for metric in ("map50", "map"):
            cand = [r for r in rows if r["name"] not in ("always_super", "always_base")
                    and (r.get("family") or r.get("kind", "")).startswith("policy")
                    and r[metric] >= sup[metric] - 1e-9]
            if not cand:
                continue
            b = min(cand, key=lambda r: r["gflops"])
            dfl = (b["gflops"] - sup["gflops"]) / sup["gflops"] * 100
            dfp = (b["fps"] - sup["fps"]) / sup["fps"] * 100
            den = (b["energy_mj"] - sup["energy_mj"]) / sup["energy_mj"] * 100
            sm.append(f"| {metric.upper()} | {b['name']} | {b['super_rate']*100:.1f} | "
                      f"{b['gflops']:.2f} | {b[metric]:.4f} | {b[metric]-sup[metric]:+.4f} | "
                      f"{b['fps']:.1f} | {dfl:+.1f} | {dfp:+.1f} | {den:+.1f} |")
        sum_path = out.with_name(out.stem + "_summary.md")
        sum_path.write_text("\n".join(sm))
        print(f"[*] summary -> {sum_path}")


if __name__ == "__main__":
    main()
