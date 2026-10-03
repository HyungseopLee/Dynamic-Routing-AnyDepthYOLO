"""Per-class AP/AR at matched SUPER usage, split into safety-critical and other.

Reviewer comment: a global mAP can hide safety-critical, single-frame missed
detections. This answers it by reporting every class separately instead of the mean,
at operating points where the SUPER usage is exactly 10%, 20%, ... (solved offline
from a frame dump by step3_eval/ablation/tau_sweep.py).

Reading the table: always-base and always-super bound what any routing policy can
reach for a class, because routing only ever picks between those two executions of
the same frozen detector. A class whose two bounds nearly coincide cannot be helped
or harmed by path selection, whatever the policy.

Usage (run from repo root):
    python -m step3_eval.ablation.per_class_table --dataset kitti
    python -m step3_eval.ablation.per_class_table --dataset kitti --latex
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from step3_eval.ablation.tau_sweep import load, decide, solve_tau, score

DUMP = {"kitti": "results/step3_eval/kitti/eval/frames.npz",
        "bdd100k": "results/step3_eval/bdd100k/eval/frames.npz",
        "waymo": "results/step3_eval/waymo/eval_both/frames.npz"}

# names and the safety-critical split: vulnerable road users, i.e. the classes a
# missed detection endangers directly
CLASSES = {
    "kitti": {0: "car", 1: "van", 2: "truck", 3: "pedestrian", 4: "person_sitting",
              5: "cyclist", 6: "tram"},
    "bdd100k": {0: "pedestrian", 1: "rider", 2: "car", 3: "bus", 4: "truck",
                5: "bicycle", 6: "motorcycle", 9: "train"},
    "waymo": {0: "vehicle", 1: "pedestrian", 2: "cyclist"},
}
SAFETY = {"kitti": {3, 4, 5}, "bdd100k": {0, 1, 5, 6}, "waymo": {1, 2}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti", choices=list(DUMP))
    ap.add_argument("--dump", default=None)
    ap.add_argument("--targets", default="10,30,50,70,90")
    ap.add_argument("--metric", default="both", choices=["ap", "ar", "both"])
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load(args.dump or DUMP[args.dataset])
    names, safe = CLASSES[args.dataset], SAFETY[args.dataset]
    present = [int(c) for c in d["gt_cls"]]
    cols = ([c for c in present if c in safe], [c for c in present if c not in safe])
    n_gt = dict(zip(present, d["gt_count"].sum(0).tolist()))

    rows = []
    for t in [0.0] + [float(x) / 100 for x in args.targets.split(",")] + [1.0]:
        per_seed = []
        for tag in d["tags"]:
            if t in (0.0, 1.0):
                take = np.full(len(d["seq"]), t == 1.0)
            else:
                tau, _ = solve_tau(d, tag, t)
                take = decide(d, tag, tau)
            per_seed.append((take.mean(), score(d, take, per_class=True)[3]))
            if t in (0.0, 1.0):
                break
        label = ("always-base" if t == 0 else "always-super" if t == 1
                 else f"{t*100:.0f}%")
        # dataset_map_ar_compact keys per_class by int (the JSON round-trip in
        # eval_video.py is what turns them into strings)
        pc = {c: {m: float(np.mean([s[1][c][m] for s in per_seed])) * 100
                  for m in ("ap", "ar")} for c in present}
        rows.append({"label": label, "super": float(np.mean([s[0] for s in per_seed])) * 100,
                     "per_class": pc})

    for metric in (["ap", "ar"] if args.metric == "both" else [args.metric]):
        title = "AP@[.5:.95]" if metric == "ap" else "AR@100"
        print(f"\n=== {args.dataset}  {title} ===")
        head = f"{'routing':<13}" + "".join(f"{names[c]:>14}" for c in cols[0]) \
               + "  |" + "".join(f"{names[c]:>14}" for c in cols[1])
        print(head)
        print(f"{'(#GT)':<13}" + "".join(f"{n_gt[c]:>14,}" for c in cols[0])
              + "  |" + "".join(f"{n_gt[c]:>14,}" for c in cols[1]))
        print("-" * len(head))
        for r in rows:
            print(f"{r['label']:<13}"
                  + "".join(f"{r['per_class'][c][metric]:>14.2f}" for c in cols[0])
                  + "  |" + "".join(f"{r['per_class'][c][metric]:>14.2f}" for c in cols[1]))
        b, s = rows[0]["per_class"], rows[-1]["per_class"]
        print(f"{'range':<13}"
              + "".join(f"{s[c][metric]-b[c][metric]:>+14.2f}" for c in cols[0])
              + "  |" + "".join(f"{s[c][metric]-b[c][metric]:>+14.2f}" for c in cols[1]))

    if args.latex:
        print("\n" + latex(args.dataset, rows, cols, names))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\n[*] {args.out}")


def latex(ds, rows, cols, names):
    sc, ot = cols
    spec = "l" + "r" * len(sc) + "r" * len(ot)
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule",
           f"& \\multicolumn{{{len(sc)}}}{{c}}{{safety-critical}} & "
           f"\\multicolumn{{{len(ot)}}}{{c}}{{other}}\\\\",
           f"\\cmidrule(lr){{2-{1+len(sc)}}}\\cmidrule(lr){{{2+len(sc)}-{1+len(sc)+len(ot)}}}",
           "Routing & " + " & ".join(names[c] for c in sc + ot) + r"\\", r"\midrule"]
    for m, tag in (("ap", "AP@[.5:.95]"), ("ar", "AR@100")):
        out.append(f"\\multicolumn{{{1+len(sc)+len(ot)}}}{{l}}{{\\textit{{{tag}}}}}\\\\")
        for r in rows:
            out.append(r["label"] + " & " + " & ".join(
                f"{r['per_class'][c][m]:.2f}" for c in sc + ot) + r"\\")
        out.append(r"\midrule" if m == "ap" else r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


if __name__ == "__main__":
    main()
