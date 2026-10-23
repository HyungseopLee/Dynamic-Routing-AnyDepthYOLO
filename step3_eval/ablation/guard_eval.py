"""Bolt Glimpse's scene-change trigger onto an already-tuned router: what happens?

Design: pick the router threshold tau that hits a target SUPER usage (e.g. 10%)
WITHOUT the guard. Then turn the guard on, reusing that SAME tau -- exactly as a
deployment would if this were added as a safety net on top of an already-tuned
system. The guard forces extra frames to SUPER on top of whatever tau already
selects, so SUPER usage goes up; we report by how much, plainly, alongside the
resulting AP/AR change. Nothing is normalised away.

    tau0 = solve_tau(target usage, no guard)
    before: decide(tau0)              -> usage_before = target,      AP/AR before
    after:  decide(tau0, guard=g)      -> usage_after >= usage_before, AP/AR after

The guard trigger: Glimpse [SenSys'15] Eq. 1-3, changed-pixel fraction (frame_
features.py) > phi_motion. phi_motion=0.5 is Glimpse's own rule (half the frame's
pixels); lower values test a more sensitive detector than the paper specifies.

Usage (run from repo root):
    python -m step3_eval.ablation.guard_eval \
        --dump results/step3_eval/kitti/eval/frames.npz \
        --features results/step3_eval/kitti/eval/frame_features.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from step3_eval.ablation.tau_sweep import load, decide, solve_tau, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--targets", default="10,30,50,70,90")
    ap.add_argument("--thr", default="0.5,0.4,0.3,0.2,0.1",
                    help="comma-sep phi_motion values. 0.5 is Glimpse's own rule")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load(args.dump)
    z = np.load(args.features, allow_pickle=False)
    targets = [float(t) / 100 for t in args.targets.split(",")]
    thrs = [float(x) for x in args.thr.split(",")]
    tags = d["tags"]

    out_thrs = []
    for thr in thrs:
        g = z["glimpse"] > thr
        tag_ = "Glimpse's phi_motion" if thr == 0.5 else "lowered"
        print(f"\nphi_motion={thr:g} ({tag_}): fires on {int(g.sum())} / {len(g)} "
              f"frames ({g.mean()*100:.1f}%)", flush=True)

        rows = []
        for t in targets:
            before, after, on_trigger_before, on_trigger_after = [], [], [], []
            for tg in tags:
                tau0, usage0 = solve_tau(d, tg, t)
                take_before = decide(d, tg, tau0)
                take_after = decide(d, tg, tau0, guard=g)
                before.append((take_before.mean(),) + score(d, take_before))
                after.append((take_after.mean(),) + score(d, take_after))
                # Scored using ONLY the frames the trigger fired on, so the effect
                # of forcing SUPER there is not diluted by the rest of the dataset
                # (which is identical before/after -- the guard only touches g).
                on_trigger_before.append(score(d, take_before, subset=g))
                on_trigger_after.append(score(d, take_after, subset=g))
            b = np.array(before); a = np.array(after)
            ob = np.array(on_trigger_before); oa = np.array(on_trigger_after)
            usage_after = float(a[:, 0].mean())

            # Fair (equal-compute) comparison: what would the router alone score if
            # it simply used the SAME final usage the guard ended up spending? This
            # isolates whether the guard's frame selection is actually good, as
            # opposed to the AP/AR change merely reflecting extra compute -- any
            # policy that spends more of the budget tends to score higher.
            matched = []
            for tg in tags:
                tau_m, _ = solve_tau(d, tg, usage_after)
                matched.append(score(d, decide(d, tg, tau_m)))
            m = np.array(matched)

            row = {"target": t * 100,
                   "usage_before": float(b[:, 0].mean()) * 100,
                   "usage_after": usage_after * 100,
                   "ap_before": float(b[:, 2].mean()) * 100,
                   "ap_after": float(a[:, 2].mean()) * 100,
                   "ar_before": float(b[:, 3].mean()) * 100,
                   "ar_after": float(a[:, 3].mean()) * 100,
                   "ap_matched": float(m[:, 1].mean()) * 100,
                   "ar_matched": float(m[:, 2].mean()) * 100}
            row["d_usage"] = row["usage_after"] - row["usage_before"]
            row["d_ap"] = row["ap_after"] - row["ap_before"]
            row["d_ar"] = row["ar_after"] - row["ar_before"]
            row["verdict_ap"] = row["ap_after"] - row["ap_matched"]
            row["verdict_ar"] = row["ar_after"] - row["ar_matched"]
            row["ap_on_trigger_before"] = float(ob[:, 1].mean()) * 100 if g.sum() else float("nan")
            row["ap_on_trigger_after"] = float(oa[:, 1].mean()) * 100 if g.sum() else float("nan")
            row["ar_on_trigger_before"] = float(ob[:, 2].mean()) * 100 if g.sum() else float("nan")
            row["ar_on_trigger_after"] = float(oa[:, 2].mean()) * 100 if g.sum() else float("nan")
            rows.append(row)
            trig_str = (f"on-trigger AP {row['ap_on_trigger_before']:.2f}->"
                       f"{row['ap_on_trigger_after']:.2f} AR "
                       f"{row['ar_on_trigger_before']:.2f}->"
                       f"{row['ar_on_trigger_after']:.2f}" if g.sum() else "no frames fired")
            print(f"  target {t*100:>3.0f}%: usage {row['usage_before']:.1f}%->"
                  f"{row['usage_after']:.1f}% ({row['d_usage']:+.1f}pp)  "
                  f"AP {row['ap_before']:.2f}->{row['ap_after']:.2f} "
                  f"(vs router-alone at same usage: {row['ap_matched']:.2f}, "
                  f"verdict {row['verdict_ap']:+.3f})  "
                  f"AR verdict {row['verdict_ar']:+.3f}  | {trig_str}", flush=True)
        out_thrs.append({"thr": thr, "fires": int(g.sum()), "n_frames": len(g),
                         "rows": rows})

    if args.out:
        Path(args.out).write_text(json.dumps({"thresholds": out_thrs}, indent=2))
        print(f"\n[*] {args.out}")


if __name__ == "__main__":
    main()
