"""Re-tabulate Table 3 (tab:main) on a UNIFIED operating-point grid so every dataset
has the same number of rows. Each sub-table reports SUPER usage at the fixed targets
{0, 10, 20, ..., 100}% (2 anchors + 9 interior). The threshold tau that realizes each
target differs per dataset (because the advantage distribution differs) and is shown
in the first column, interpolated from that dataset's video Pareto curve.

Latency/FPS/energy reuse the device anchors already measured by bench_device.py
(results/step3_eval/device_table.json); AP and GFLOPs are interpolated from the curve at each
target SUPER% (both are device-independent). No GPU needed.

    python -m analysis.unify_device_table
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/step3_eval"
TARGETS = [10, 20, 30, 40, 50, 60, 70, 80, 90]   # interior; 0/100 are the anchors

CURVES = {
    "KITTI":   (OUT / "kitti/eval/video_curve_main_both_g2.json", r"router_both_s(\d+)"),
    "BDD100K": (OUT / "bdd100k/eval/video_curve_full_aligned.json", r"router_bn(\d+)"),
    "Waymo":   (OUT / "waymo/eval_both/video_curve.json", r"router_seed(\d+)"),
}


def curve_arrays(curve_json, pol_re, metric="map"):
    """Return (super%, thres, gflops, AP%) arrays for the router sweep, averaged over
    seeds and sorted by SUPER usage, plus the two anchors."""
    d = json.loads(Path(curve_json).read_text())
    pol_re = re.compile(pol_re)
    ends, pol = {}, defaultdict(list)
    for r in d["rows"]:
        if r["name"] in ("always_base", "always_super"):
            ends[r["name"]] = (r["super_rate"] * 100, r["gflops"], r[metric] * 100)
        elif pol_re.fullmatch(r.get("family", "")):
            k = r["budget"] if r.get("budget") is not None else r["thres"]
            pol[k].append((r["super_rate"] * 100, r["thres"], r["gflops"], r[metric] * 100))
    pts = [np.array(pol[k]).mean(0) for k in pol]
    pts = np.array(sorted(pts, key=lambda p: p[0]))
    # AP / GFLOPs interpolation spans [0,100] with the (real, measured) anchors.
    sb, gb, ab = ends["always_base"]; ss, gs, asu = ends["always_super"]
    sup = np.concatenate(([sb], pts[:, 0], [ss]))
    gfl = np.concatenate(([gb], pts[:, 2], [gs]))
    ap = np.concatenate(([ab], pts[:, 3], [asu]))
    order = np.argsort(sup); sup = sup[order]; gfl = gfl[order]; ap = ap[order]
    # tau has NO real value at the anchors (always-base/super correspond to
    # tau=+-inf). Interpolate tau ONLY over non-degenerate measured points
    # (1% < SUPER% < 99%); np.interp then CLAMPS targets outside that range to
    # the nearest measured threshold, so tau never leaves the realized range nor
    # crosses the sparse 0%->first-point gap (which produced inflated values).
    keep = (pts[:, 0] > 1.0) & (pts[:, 0] < 99.0)
    sup_t, thr_t = pts[keep, 0], pts[keep, 1]
    o2 = np.argsort(sup_t); sup_t, thr_t = sup_t[o2], thr_t[o2]
    return sup, gfl, ap, ends, sup_t, thr_t


def main():
    dev = json.loads((OUT / "device_table.json").read_text())
    for name, (curve, pol_re) in CURVES.items():
        if name not in dev:
            print(f"% [!] {name}: no device anchors; skip"); continue
        v = dev[name]
        lb, ls, eb, es, rt = v["lat_base"], v["lat_super"], v["eng_base"], v["eng_super"], v["router_ms"]
        sup, gfl, ap, ends, sup_t, thr_t = curve_arrays(curve, pol_re)

        def blend(p):                       # p in [0,1]
            lat = (1 - p) * lb + p * ls + rt
            return lat, 1000.0 / lat, (1 - p) * eb + p * es

        def row(label, sp, gf, apv, router=True):
            # static single-path anchors run WITHOUT the router (no rt); routed
            # operating points pay the router overhead on every frame.
            p = sp / 100.0
            lat = (1 - p) * lb + p * ls + (rt if router else 0.0)
            en = (1 - p) * eb + p * es
            return (f"{label} & {sp:.0f} & {gf:.2f} & {apv:.1f} & "
                    f"{lat:.2f} & {1000.0/lat:.1f} & {en:.0f}\\\\")

        print(f"\n% ---- {name} (unified 0..100 grid) ----")
        gb, ab = ends["always_base"][1], ends["always_base"][2]
        gs, asu = ends["always_super"][1], ends["always_super"][2]
        print("  " + row(r"\textsc{always-base}", 0, gb, ab, router=False))
        for t in TARGETS:
            tau = float(np.interp(t, sup_t, thr_t))   # clamped to measured tau range
            gf = float(np.interp(t, sup, gfl))
            apv = float(np.interp(t, sup, ap))
            print("  " + row(f"${tau:+.3f}$", t, gf, apv))
        print("  " + row(r"\textsc{always-super}", 100, gs, asu, router=False))


if __name__ == "__main__":
    main()
