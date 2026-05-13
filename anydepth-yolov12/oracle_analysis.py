"""Oracle analysis for AnyDepth router.

Bounds the achievable Pareto frontier using cached per-image (L_super, L_base):
  - Always-SUPER baseline
  - Always-BASE baseline
  - Oracle binary (per-image best of BASE/SUPER)
  - Distribution of (L_base - L_super) to characterize signal

Usage: python oracle_analysis.py [--cache-dir PATH]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def L_to_mAP_linear(L, L_super_mean, L_base_mean, mAP_super=35.0, mAP_base=34.1):
    """Rough linear proxy: 35.0 at L_super_mean, 34.1 at L_base_mean."""
    return mAP_super - (L - L_super_mean) / (L_base_mean - L_super_mean) * (mAP_super - mAP_base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="./runs/bdd100k/detect/anydepth-yolov12s/"
                        "50e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_orig/router/cache")
    parser.add_argument("--mAP-super", type=float, default=35.0)
    parser.add_argument("--mAP-base",  type=float, default=34.1)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache = torch.load(cache_dir / "cache.pt", map_location="cpu", weights_only=False)
    flops_meta = json.load(open(cache_dir / "flops_meta.json"))

    L_super = cache["L_super"].float()  # [N]
    L_base  = cache["L_base"].float()
    F_super = float(flops_meta["F_super"])  # FLOPs (= 2 * MACs)
    F_base  = float(flops_meta["F_base"])
    N = L_super.shape[0]

    diff = L_base - L_super  # >0 => SUPER better, <0 => BASE better (inverted)

    L_super_mean = L_super.mean().item()
    L_base_mean  = L_base.mean().item()

    print("=" * 78)
    print(f"[ORACLE ANALYSIS]   N = {N} images")
    print("=" * 78)

    # ---- 1. Reference points ----
    print("\n--- 1. Reference points ---")
    print(f"{'Strategy':<22}{'L_mean':>10}{'F_mean (G)':>14}{'mAP (cal)':>12}")
    print("-" * 58)
    print(f"{'always-SUPER':<22}{L_super_mean:>10.4f}{F_super/1e9:>14.2f}{args.mAP_super:>12.2f}")
    print(f"{'always-BASE':<22}{L_base_mean:>10.4f}{F_base/1e9:>14.2f}{args.mAP_base:>12.2f}")

    # ---- 2. ΔL distribution ----
    print("\n--- 2. ΔL distribution (L_base - L_super) ---")
    print(f"  mean  = {diff.mean():+.4f}")
    print(f"  std   = {diff.std():.4f}")
    print(f"  min   = {diff.min():+.4f}")
    print(f"  max   = {diff.max():+.4f}")

    n_inv = int((diff < 0).sum().item())
    print(f"\n  Inverted (BASE better): {n_inv}/{N} ({100*n_inv/N:.2f}%)")

    print(f"\n  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = float(np.percentile(diff.numpy(), p))
        print(f"    p{p:>3d} = {val:+.4f}")

    # ---- 3. Histogram (text bars) ----
    print(f"\n  Distribution (% of samples):")
    edges = [-2.0, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        cnt = int(((diff >= lo) & (diff < hi)).sum().item())
        pct = 100 * cnt / N
        bar = "#" * int(round(pct / 2))
        print(f"    [{lo:+.2f}, {hi:+.2f}): {pct:5.2f}%  {bar}")

    # ---- 4. Oracle binary ----
    print("\n--- 3. Oracle binary (per-image best of BASE/SUPER) ---")
    pick_super = (diff > 0)  # SUPER chosen when it has lower loss
    oracle_L = torch.where(pick_super, L_super, L_base)
    oracle_F = torch.where(pick_super, torch.tensor(F_super), torch.tensor(F_base))
    n_super = int(pick_super.sum().item())
    n_base  = N - n_super

    oracle_L_mean = oracle_L.mean().item()
    oracle_F_mean = oracle_F.mean().item()
    oracle_mAP = L_to_mAP_linear(oracle_L_mean, L_super_mean, L_base_mean,
                                  args.mAP_super, args.mAP_base)

    print(f"  Picked SUPER: {n_super}/{N} ({100*n_super/N:.2f}%)")
    print(f"  Picked BASE : {n_base}/{N} ({100*n_base/N:.2f}%)")
    print(f"  Oracle L_mean = {oracle_L_mean:.4f}")
    print(f"  Oracle F_mean = {oracle_F_mean/1e9:.2f} G")
    print(f"  Oracle mAP (rough proxy) = {oracle_mAP:.3f}")

    # ---- 5. Comparison ----
    print("\n--- 4. Comparison ---")
    print(f"{'':22}{'L_mean':>10}{'F (G)':>10}{'mAP':>10}{'ΔF vs SUP':>14}{'ΔmAP vs SUP':>14}")
    print("-" * 80)
    rows = [
        ("always-BASE",  L_base_mean,  F_base,  args.mAP_base),
        ("Oracle binary", oracle_L_mean, oracle_F_mean, oracle_mAP),
        ("always-SUPER", L_super_mean, F_super, args.mAP_super),
    ]
    for name, L, F, mAP in rows:
        dF  = (F - F_super) / 1e9
        dmAP = mAP - args.mAP_super
        print(f"{name:<22}{L:>10.4f}{F/1e9:>10.2f}{mAP:>10.3f}"
              f"{dF:>+14.2f}{dmAP:>+14.3f}")

    # ---- 6. Summary verdict ----
    super_savings_F = (F_super - oracle_F_mean) / F_super
    super_loss_mAP  = args.mAP_super - oracle_mAP
    print("\n--- 5. Verdict (Oracle binary vs always-SUPER) ---")
    print(f"  FLOPs savings  : {100 * super_savings_F:.2f} %  "
          f"({(F_super - oracle_F_mean)/1e9:.2f} G saved)")
    print(f"  mAP loss       : {super_loss_mAP:+.3f}  (oracle binary - SUPER)")

    print("\n  Routing ceiling characterization:")
    if super_savings_F < 0.10 and abs(super_loss_mAP) < 0.05:
        print("    → Routing offers MINIMAL benefit. Just use SUPER.")
    elif super_savings_F < 0.20:
        print("    → Routing offers MODEST benefit. Limited ceiling for further engineering.")
    elif super_savings_F < 0.40:
        print("    → Routing offers MEANINGFUL benefit. Worth investing in better reward.")
    else:
        print("    → Routing offers SUBSTANTIAL benefit. Strong case for continuing.")

    # ---- 7. Compare to current trained policies ----
    print("\n--- 6. Where do trained policies sit on this Pareto? ---")
    print("  (mAP and F values from train logs; mAP estimated via linear proxy)")
    trained = [
        ("v1 ep9 (loss, c_e=0.10)", 42.24, -0.6139),
        ("v2 ep1 (loss, c_e=0.02)", 37.80, -0.7968),
    ]
    for name, F_pred, q in trained:
        # quality = (L_super - L_pred) / scale, scale = mean(diff)
        # so L_pred = L_super_mean - q * scale
        scale = diff.mean().item()
        L_pred_est = L_super_mean - q * scale
        mAP_est = L_to_mAP_linear(L_pred_est, L_super_mean, L_base_mean,
                                   args.mAP_super, args.mAP_base)
        print(f"    {name:<28} F={F_pred:.2f}G  L_pred≈{L_pred_est:.4f}  mAP≈{mAP_est:.3f}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
