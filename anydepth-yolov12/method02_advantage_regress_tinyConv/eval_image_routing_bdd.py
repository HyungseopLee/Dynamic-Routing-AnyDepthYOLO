"""Image-val routing curve (paper Fig 2a analog) on BDD det_20 val.

Per-image, NON-recursive protocol (exactly the paper's): for every image run BOTH
the BASE and SUPER paths, keep each path's matches, and the cheap signals
  lum(x), edge(x)            (Eq 7,8)  low -> SUPER
  conf(x) = mean top-20 BASENET score (Eq 6)  low -> SUPER
Then for each strategy, pick base/super matches per image and aggregate the true
dataset AP via B.dataset_map_multi_iou (so AP is set-level, not per-image averaged).
The learned routers (bb=input, bn=both, pn=pred) are scored from cache_val_both
features (path_id=BASE), aligned to images by im_file. Output: a curve json + a
plot identical in style to the MOT video figure, so image-val vs video are directly
comparable.

    python method02_advantage_regress_tinyConv/eval_image_routing_bdd.py \
        --weight finetuned_bdd100k/<alpha0.2>.pt --cache outputs/bdd100k/cache_val_both.pt --n 2000
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa
import eval_baseline_kitti as B  # noqa
from method02_advantage_regress_tinyConv.eval_video_bdd import (  # noqa
    BDD_MOT_EVAL_CLS, pixel_signals, load_policy)
from method02_advantage_regress_tinyConv.eval_image_protocol_bdd import load_gt  # noqa

OUT = Path(__file__).resolve().parent / "outputs"


def dataset_ap(per_img_matches, gt_count):
    flat = [m for mm in per_img_matches for m in mm]
    _, m50, m5095 = B.dataset_map_multi_iou(flat, gt_count)
    return m50, m5095


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--cache", default=str(OUT / "bdd100k/cache_val_both.pt"))
    ap.add_argument("--img_root", default="/media/data/bdd100k_yolo/val")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[720, 1280])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=2000, help="0 = all cached images")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT / "bdd100k/eval/image_curve.json"))
    args = ap.parse_args()

    B.EVAL_CLS = BDD_MOT_EVAL_CLS
    dev = args.device if torch.cuda.is_available() else "cpu"
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    files = list(cache["im_file"])
    n = len(files) if args.n <= 0 else min(args.n, len(files))

    # learned routers: tinyConv (tc, conv on 2x2 grid) and GAP-MLP (gm, MLP on GAP
    # vector). family tag -> list of (net, feat, arch) over 5 seeds.
    from method01_advantage_regress.policy_net import PolicyNetwork as GapMlp
    fams = {  # tag -> (stem, arch)
        "bb": ("policy", "tc"), "bn": ("policy_both", "tc"), "pn": ("policy_pred", "tc"),
        "gmbb": ("gapmlp_input", "gm"), "gmbn": ("gapmlp_both", "gm"), "gmpn": ("gapmlp_pred", "gm")}
    inp = cache["input_base"][:n].to(dev); prd = cache["pred_base"][:n].to(dev)
    inp_g, prd_g = inp.mean(dim=(2, 3)), prd.mean(dim=(2, 3))   # GAP vectors for gm
    pid = torch.zeros(n, dtype=torch.long, device=dev)

    def load_gm(path):
        ck = torch.load(path, map_location=dev, weights_only=False); a = ck.get("args", {})
        feat = a.get("feat", "both")
        net = GapMlp(group_dim=a.get("group_dim", 64), path_dim=a.get("path_dim", 8),
                     hidden_dim=a.get("hidden", 64), feat=feat, norm=a.get("norm", "batch")).to(dev)
        with torch.no_grad():                       # materialise LazyLinear
            net(inp_g[:2], prd_g[:2], pid[:2])
        net.load_state_dict(ck["state_dict"]); net.eval()
        return net, feat

    router_score = {}  # fam -> [nseed, n] predicted advantage
    with torch.no_grad():
        for fam, (stem, arch) in fams.items():
            sc = []
            for s in range(5):
                p = OUT / f"bdd100k/{stem}_{s}.pt"
                if not p.exists():
                    continue
                if arch == "tc":
                    net, ckpt, feat, _ = load_policy(str(p), dev)
                    net.load_state_dict(ckpt["state_dict"]); net.eval()
                    xi, xp = inp, prd
                else:
                    net, feat = load_gm(str(p)); xi, xp = inp_g, prd_g
                sc.append(net.logit(xi, None if feat == "input" else xp, pid).view(-1).cpu().numpy())
            if sc:
                router_score[fam] = np.stack(sc)

    yolo = YOLO(args.weight, task="detect")
    N = getattr(yolo.model, "num_skippable_layers", 0)
    skip = {"base": [True] * N, "super": [False] * N}

    mt = {"base": [], "super": []}      # per-image matches per path
    gt_count = defaultdict(int)
    lum, edge, conf = [], [], []
    lbl_dir = Path(args.img_root) / "labels"
    for i in range(n):
        bgr = cv2.imread(files[i])
        H, W = bgr.shape[:2]
        gts = load_gt(lbl_dir / f"{Path(files[i]).stem}.txt", W, H)
        l, e = pixel_signals(bgr); lum.append(l); edge.append(e)
        cb = None
        for tag in ("super", "base"):
            r = yolo.predict(source=bgr, imgsz=tuple(args.imgsz), conf=args.conf, iou=0.7,
                             skip=skip[tag], verbose=False, device=dev)[0]
            preds = B.boxes_to_preds(r)
            m, gtc = B.match_frame_multi_iou(preds, gts)
            mt[tag].append(m)
            if tag == "base":
                cf = r.boxes.conf.cpu() if (r.boxes is not None and len(r.boxes)) else torch.empty(0)
                cb = B.conf_top_k_mean(cf, 20) if cf.numel() else 0.0
        conf.append(cb)
        # gt_count is path-independent; accumulate once from the (identical) gts
        for g in gts:
            gt_count[g[0]] += 1
        if (i + 1) % 250 == 0:
            print(f"[*] {i+1}/{n}")
    lum, edge, conf = map(np.array, (lum, edge, conf))
    gt_count = dict(gt_count)

    ftab = json.loads((OUT / "bdd100k/flops_table.json").read_text())["actions"]
    gb, gs = ftab["0_base"]["gflops"], ftab["1_super"]["gflops"]

    def route_ap(super_mask):
        sel = [mt["super"][i] if super_mask[i] else mt["base"][i] for i in range(n)]
        m50, m5095 = dataset_ap(sel, gt_count)
        sr = float(super_mask.mean())
        return {"map50": m50, "map": m5095, "super_rate": sr,
                "gflops": sr * gs + (1 - sr) * gb}

    rows = []
    def add(name, kind, family, super_mask, budget=None):
        r = route_ap(super_mask); r.update(name=name, kind=kind, family=family, budget=budget)
        rows.append(r)

    add("always_base", "const", "always", np.zeros(n, bool))
    add("always_super", "const", "const", np.ones(n, bool))
    budgets = list(range(5, 100, 5))

    # signal routers: route the b% with LOWEST signal to SUPER (lum/edge/conf);
    # confidence direction matches: low conf -> super.
    for nm, sig in (("lum", lum), ("edge", edge), ("conftop20", conf)):
        order = np.argsort(sig)                 # lowest first -> super
        for b in budgets:
            k = int(round(b / 100 * n))
            mask = np.zeros(n, bool); mask[order[:k]] = True
            add(f"{nm}_b{b:02d}", nm, nm, mask, b)
    # random baseline (sweep): random b% to super (fixed seed for reproducibility)
    rng = np.random.default_rng(0)
    for b in budgets:
        mask = rng.random(n) < (b / 100)
        add(f"random_p{b:03d}", "random", "random", mask, b)
    # learned routers: route the b% with HIGHEST predicted advantage to super
    for fam, scores in router_score.items():
        for si in range(scores.shape[0]):
            order = np.argsort(-scores[si])     # highest A-hat first -> super
            for b in budgets:
                k = int(round(b / 100 * n))
                mask = np.zeros(n, bool); mask[order[:k]] = True
                add(f"policy_{fam}{si}_b{b:02d}", "policy", f"policy_{fam}{si}", mask, b)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows, "n": n,
        "meta": {"conf": args.conf, "imgsz": args.imgsz}}, indent=1))
    base = next(r for r in rows if r["name"] == "always_base")
    sup = next(r for r in rows if r["name"] == "always_super")
    print(f"[*] n={n} base mAP50={base['map50']:.4f} super={sup['map50']:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
