"""Merge raw per-shard outputs of eval_video_bdd.py (--raw_out) into the final
AP-vs-FLOPs table. Each shard evaluated a disjoint subset of videos with an
IDENTICAL strategy list (val-derived + global pixel thresholds), so per-strategy
matches and GT counts simply concatenate/sum before computing dataset mAP.

Memory note: at conf=0.001 each shard holds millions of per-pred match tuples
(cls, conf, [is_tp@iou x10]). Holding all shards' raw python tuples at once OOMs
(a ~4.5 GB pickle expands to tens of GB of object overhead). So we stream one
shard at a time and immediately compact each strategy's matches into three numpy
arrays (cls int16, conf float32, tp bool[N,10]), freeing the raw dict before the
next shard. Peak memory = one raw shard + the compact arrays.

Usage:
    python method02_advantage_regress_tinyConv/merge_video_shards.py \
        --shards outputs/bdd100k/eval/shard_*.pt \
        --out    outputs/bdd100k/eval/video_curve.json
"""

import argparse
import gc
import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import eval_baseline_kitti as B
from method02_advantage_regress_tinyConv.eval_video_bdd import BDD_MOT_EVAL_CLS


def compact_map_multi_iou(cls_arr, conf_arr, tp_arr, gt_count, iou_grid=B.IOU_GRID):
    """Vectorised dataset_map_multi_iou over compact arrays.
    cls_arr:int[N]  conf_arr:float[N]  tp_arr:bool[N,len(iou_grid)]."""
    order = np.argsort(-conf_arr, kind="stable")     # sort by conf desc once
    cls_s, tp_s = cls_arr[order], tp_arr[order]
    ap_iou_cls = {ti: {} for ti in range(len(iou_grid))}
    for cls in B.EVAL_CLS:
        n_gt = gt_count.get(cls, 0)
        if n_gt == 0:
            continue
        m = cls_s == cls
        if not m.any():
            for ti in range(len(iou_grid)):
                ap_iou_cls[ti][cls] = 0.0
            continue
        tp_c = tp_s[m]                                # [Nc, n_iou], conf-desc order
        for ti in range(len(iou_grid)):
            is_tp = tp_c[:, ti].astype(np.float64)
            tp_cum = np.cumsum(is_tp)
            fp_cum = np.cumsum(1.0 - is_tp)
            precs = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            recs = tp_cum / n_gt
            ap_iou_cls[ti][cls] = B.compute_ap(precs, recs)
    ap50 = ap_iou_cls[0]
    map50 = float(np.mean(list(ap50.values()))) if ap50 else 0.0
    per_iou = [float(np.mean(list(ap_iou_cls[ti].values()))) if ap_iou_cls[ti] else 0.0
               for ti in range(len(iou_grid))]
    return ap50, map50, float(np.mean(per_iou))


def main():
    B.EVAL_CLS = BDD_MOT_EVAL_CLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="raw shard .pt files (globs ok)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted({p for g in args.shards for p in glob.glob(g)})
    if not paths:
        raise SystemExit("no shard files matched")
    print(f"[*] merging {len(paths)} shards (low-mem streaming)")

    n_iou = len(B.IOU_GRID)
    # per strategy: lists of compact per-shard arrays (concatenated at the end)
    cls_parts = defaultdict(list)
    conf_parts = defaultdict(list)
    tp_parts = defaultdict(list)
    gt_count = defaultdict(lambda: defaultdict(int))
    n_super = defaultdict(int)
    n_base = defaultdict(int)
    meta = {}
    gs = gb = None
    total_frames = 0

    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        gs, gb = d["gflops_super"], d["gflops_base"]
        total_frames += d.get("total_frames", 0)
        meta.update(d["meta"])
        for name, st in d["state"].items():
            mm = st["matches_multi"]
            if mm:
                cls_parts[name].append(np.fromiter((t[0] for t in mm), dtype=np.int16, count=len(mm)))
                conf_parts[name].append(np.fromiter((t[1] for t in mm), dtype=np.float32, count=len(mm)))
                tp = np.empty((len(mm), n_iou), dtype=bool)
                for i, t in enumerate(mm):
                    tp[i] = t[2]
                tp_parts[name].append(tp)
            for k, v in st["gt_count"].items():
                gt_count[name][k] += v
            n_super[name] += st["n_super"]
            n_base[name] += st["n_base"]
        del d
        gc.collect()
        print(f"[*] folded {Path(p).name}")

    rows = []
    for name in meta:
        cls_arr = np.concatenate(cls_parts[name]) if cls_parts[name] else np.empty(0, np.int16)
        conf_arr = np.concatenate(conf_parts[name]) if conf_parts[name] else np.empty(0, np.float32)
        tp_arr = np.concatenate(tp_parts[name]) if tp_parts[name] else np.empty((0, n_iou), bool)
        _, map50, map5095 = compact_map_multi_iou(cls_arr, conf_arr, tp_arr, gt_count[name])
        # free this strategy's arrays before the next
        cls_parts[name] = conf_parts[name] = tp_parts[name] = None
        n = n_super[name] + n_base[name]
        super_rate = n_super[name] / max(n, 1)
        gflops = super_rate * gs + (1 - super_rate) * gb
        if "_t" in name:
            family = name.rsplit("_t", 1)[0]
        elif "_b" in name:
            family = name.rsplit("_b", 1)[0]
        else:
            family = meta[name]["kind"]
        rows.append({"name": name, "kind": meta[name]["kind"], "family": family,
                     "thres": meta[name]["thres"], "budget": meta[name].get("budget"),
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
        f.write(f"# merge_video_shards.py {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# shards={len(paths)} frames={total_frames} strategies={len(rows)} "
                f"gflops_base={gb:.2f} gflops_super={gs:.2f}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"[*] merged -> {out}\n[*] log -> {out.with_suffix('.log')}")


if __name__ == "__main__":
    main()
