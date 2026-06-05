"""Merge raw per-shard outputs of eval_video_bdd.py (--raw_out) into the final
AP-vs-FLOPs table. Each shard evaluated a disjoint subset of videos with an
IDENTICAL strategy list (val-derived + global pixel thresholds), so per-strategy
matches and GT counts simply concatenate/sum before computing dataset mAP.

Usage:
    python method02_advantage_regress_tinyConv/merge_video_shards.py \
        --shards outputs/bdd100k/eval/shard_*.pt \
        --out    outputs/bdd100k/eval/video_curve.json
"""

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import eval_baseline_kitti as B
from method02_advantage_regress_tinyConv.eval_video_bdd import BDD_MOT_EVAL_CLS


def main():
    B.EVAL_CLS = BDD_MOT_EVAL_CLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="raw shard .pt files (globs ok)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted({p for g in args.shards for p in glob.glob(g)})
    if not paths:
        raise SystemExit("no shard files matched")
    print(f"[*] merging {len(paths)} shards")

    merged = {}   # name -> {matches_multi, gt_count, n_super, n_base}
    meta = {}
    gs = gb = None
    total_frames = 0
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        gs, gb = d["gflops_super"], d["gflops_base"]
        total_frames += d.get("total_frames", 0)
        meta.update(d["meta"])
        for name, st in d["state"].items():
            m = merged.setdefault(name, {"matches_multi": [], "gt_count": defaultdict(int),
                                         "n_super": 0, "n_base": 0})
            m["matches_multi"].extend(st["matches_multi"])
            for k, v in st["gt_count"].items():
                m["gt_count"][k] += v
            m["n_super"] += st["n_super"]; m["n_base"] += st["n_base"]

    rows = []
    for name, m in merged.items():
        _, map50, map5095 = B.dataset_map_multi_iou(m["matches_multi"], m["gt_count"])
        n = m["n_super"] + m["n_base"]
        super_rate = m["n_super"] / max(n, 1)
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
