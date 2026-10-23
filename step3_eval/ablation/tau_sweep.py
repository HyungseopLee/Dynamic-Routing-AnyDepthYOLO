"""Solve for the routing threshold that realises an exact SUPER usage, then score it.

The paper's main table reports a fixed threshold tau per operating point. Setting tau
from a quantile of the validation advantage distribution does not land on round SUPER
usages on the evaluation video (the two distributions differ), and the realised usage
of a candidate tau can only be known by replaying the recursive routing over the whole
video. Doing that on the GPU once per candidate would be prohibitive.

step3_eval/eval_video.py --frame_dump writes, for every frame, the matched detections
of BOTH paths and the router outputs. Replaying the recursion for a candidate tau is
then a pure numpy pass, so this script bisects tau to hit 10%, 20%, ... exactly and
pools the corresponding per-frame rows to compute AP/AR.

Usage (run from repo root):
    # 1. one GPU pass that records both paths
    python -m step3_eval.eval_video --dataset kitti ... \
        --frame_dump results/step3_eval/kitti/eval/frames.npz

    # 2. exact tau per target usage, offline
    python -m step3_eval.ablation.tau_sweep \
        --dump results/step3_eval/kitti/eval/frames.npz --targets 10,20,30,40,50,60,70,80,90
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import step3_eval.eval_utils as B


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(np.load(path, allow_pickle=False)["meta"][0]))
    for p in ("base", "super"):
        d[f"off_{p}"] = np.concatenate(([0], np.cumsum(d[f"n_{p}"])))
    d["tags"] = sorted({k[3:-5] for k in d if k.startswith("ah_") and k.endswith("_base")})
    return d


def decide(d, tag, tau, causal=True, guard=None, shuffle_seed=None):
    """Replay the recursion; returns a bool array, True where SUPER is chosen.

    `guard` is an optional per-frame boolean mask that forces SUPER regardless of the
    router, i.e. the scene-change override. Because the forced frames consume budget,
    tau is re-solved so that the TOTAL usage still hits the target: the guard and the
    router are then compared at equal compute rather than the guard simply spending
    more.

    `shuffle_seed`, if given, walks each sequence in a random order instead of
    chronological order -- reproducing step3_eval/eval_video.py --shuffle_frames
    entirely offline, since shuffling only changes which frame's advantage feeds the
    next DECISION, not which frame gets scored (each frame is still matched against
    its own ground truth via its own absolute index).
    """
    ab, asu = d[f"ah_{tag}_base"], d[f"ah_{tag}_super"]
    seq = d["seq"]
    take = np.zeros(len(seq), bool)
    starts = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1]])
    ends = np.r_[starts[1:], len(seq)]
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    for s, e in zip(starts, ends):
        order = rng.permutation(np.arange(s, e)) if rng is not None else np.arange(s, e)
        prev_val, prev_choice = 0.0, 0          # 0 = base, 1 = super
        for pos, i in enumerate(order):
            if causal:
                sup = (pos > 0) and (prev_val > tau)
                if guard is not None and guard[i]:
                    sup = True
                take[i] = sup
                prev_val = (asu if sup else ab)[i]
            else:
                pv = (asu if prev_choice else ab)[i]
                sup = (pos > 0) and (pv > tau)
                if guard is not None and guard[i]:
                    sup = True
                take[i] = sup
                prev_choice = int(sup)
    return take


def solve_tau(d, tag, target, causal=True, lo=-5.0, hi=5.0, iters=60, guard=None,
             shuffle_seed=None):
    """Bisect tau so the realised SUPER usage matches `target` (a fraction)."""
    f = lambda t: decide(d, tag, t, causal, guard, shuffle_seed).mean()
    if f(lo) < target:                      # even the loosest threshold cannot reach it
        return lo, f(lo)
    if f(hi) > target:
        return hi, f(hi)
    for _ in range(iters):                  # usage decreases as tau grows
        mid = 0.5 * (lo + hi)
        if f(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo, f(lo)


def random_take(n_frames, seq, target, rng):
    """Random routing at exactly `target` usage, drawn independently per sequence."""
    take = np.zeros(n_frames, bool)
    starts = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1]])
    ends = np.r_[starts[1:], n_frames]
    for s, e in zip(starts, ends):
        k = int(round((e - s) * target))
        if k:
            take[s + rng.permutation(e - s)[:k]] = True
    return take


def score(d, take, per_class=False, subset=None):
    """Pool the chosen path's rows per frame and return (map50, map5095, ar5095).

    With per_class=True a fourth element is returned: {class id: {ap, ap50, ar}}.
    `subset`, if given, restricts BOTH the scored frames and the GT count to that
    boolean mask -- e.g. only the frames a scene-change trigger fired on, so the
    effect of forcing SUPER there is not diluted by the rest of the dataset.
    """
    if subset is None:
        subset = np.ones(len(take), bool)
    cls, conf, tp = [], [], []
    for p, sel in (("base", (~take) & subset), ("super", take & subset)):
        off, c, cf, t = d[f"off_{p}"], d[f"cls_{p}"], d[f"conf_{p}"], d[f"tp_{p}"]
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        rng = np.concatenate([np.arange(off[i], off[i + 1]) for i in idx]) if idx.size else []
        rng = np.asarray(rng, dtype=np.int64)
        cls.append(c[rng]); conf.append(cf[rng]); tp.append(t[rng])
    cls = np.concatenate(cls) if cls else np.empty(0, np.int16)
    conf = np.concatenate(conf) if conf else np.empty(0, np.float32)
    tp = np.concatenate(tp) if tp else np.empty((0, 10), bool)
    gt = {int(c): int(n) for c, n in zip(d["gt_cls"], d["gt_count"][subset].sum(0))}
    B.EVAL_CLS = sorted(gt)
    _, map50, map5095, ar, pc = B.dataset_map_ar_compact(cls, conf, tp, gt)
    if per_class:
        return map50, map5095, ar, pc
    return map50, map5095, ar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--targets", default="10,20,30,40,50,60,70,80,90")
    ap.add_argument("--non_causal", action="store_true")
    ap.add_argument("--features", default=None,
                    help="frame_features.npz; with --guard_feature/--guard_pct a "
                         "scene-change override is applied and tau is re-solved so the "
                         "total SUPER usage still matches the target")
    ap.add_argument("--guard_feature", default="pixel",
                    choices=["pixel", "edge", "corner", "area"])
    ap.add_argument("--guard_pct", type=float, default=99.0,
                    help="percentile of the feature distribution above which SUPER is forced")
    ap.add_argument("--random_seeds", type=int, default=5,
                    help="seeds for the random baseline, evaluated offline at the same "
                         "exact usages (0 to skip)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load(args.dump)
    causal = not args.non_causal
    guard = None
    if args.features:
        z = np.load(args.features, allow_pickle=False)
        v = z[args.guard_feature]
        if len(v) != len(d["seq"]):
            raise SystemExit(f"feature file has {len(v)} frames, dump has {len(d['seq'])}")
        thr = float(np.percentile(v[1:], args.guard_pct))
        guard = v > thr
        print(f"[*] guard: {args.guard_feature} > {thr:.4f} "
              f"(top {100 - args.guard_pct:g}%) fires on {int(guard.sum())} frames")
    gb, gs = d["meta"]["gflops_base"], d["meta"]["gflops_super"]
    print(f"[*] {len(d['seq'])} frames, seeds {d['tags']}, causal={causal}")

    rows = []
    for target in [0.0] + [float(t) / 100 for t in args.targets.split(",")] + [1.0]:
        per_seed = []
        for tag in d["tags"]:
            if target in (0.0, 1.0):
                take = np.full(len(d["seq"]), target == 1.0)
                tau = float("nan")
            else:
                tau, _ = solve_tau(d, tag, target, causal, guard=guard)
                take = decide(d, tag, tau, causal, guard)
            m50, m, ar = score(d, take)
            per_seed.append((tau, take.mean(), m50, m, ar))
            if target in (0.0, 1.0):
                break
        a = np.array(per_seed, float)
        rows.append({"target": target * 100, "kind": "router",
                     "tau": float(np.nanmean(a[:, 0])),
                     "tau_std": float(np.nanstd(a[:, 0])),
                     "super": float(a[:, 1].mean()) * 100,
                     "map50": float(a[:, 2].mean()) * 100,
                     "map": float(a[:, 3].mean()) * 100,
                     "map_std": float(a[:, 3].std()) * 100,
                     "ar": float(a[:, 4].mean()) * 100,
                     "ar_std": float(a[:, 4].std()) * 100,
                     "gflops": float(a[:, 1].mean()) * gs + (1 - float(a[:, 1].mean())) * gb})

        if args.random_seeds and target not in (0.0, 1.0):
            rs = []
            for si in range(args.random_seeds):
                tk = random_take(len(d["seq"]), d["seq"], target, np.random.default_rng(si))
                m50, m, ar = score(d, tk)
                rs.append((tk.mean(), m50, m, ar))
            r = np.array(rs, float)
            rows.append({"target": target * 100, "kind": "random",
                         "super": float(r[:, 0].mean()) * 100,
                         "map50": float(r[:, 1].mean()) * 100,
                         "map": float(r[:, 2].mean()) * 100,
                         "map_std": float(r[:, 2].std()) * 100,
                         "ar": float(r[:, 3].mean()) * 100,
                         "ar_std": float(r[:, 3].std()) * 100})

    print(f"\n{'target':>12}{'tau':>10}{'Super %':>10}{'GFLOPs':>9}{'AP':>8}{'AR':>8}")
    for r in rows:
        if r["kind"] == "random":
            print(f"{'  (random)':>12}{'  --  ':>10}{r['super']:>10.1f}{'':>9}"
                  f"{r['map']:>8.2f}{r['ar']:>8.2f}")
            continue
        name = "always-base" if r["target"] == 0 else (
            "always-super" if r["target"] == 100 else f"{r['target']:.0f}")
        tau = "  --  " if np.isnan(r["tau"]) else f"{r['tau']:+.4f}"
        print(f"{name:>12}{tau:>10}{r['super']:>10.1f}{r['gflops']:>9.2f}"
              f"{r['map']:>8.2f}{r['ar']:>8.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\n[*] {args.out}")


if __name__ == "__main__":
    main()
