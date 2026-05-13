"""
Policy learning for AnyDepth-YOLO depth-config decision network.

Stages:
  --stage cache : precompute (state, L_super, L_base) per image + FLOPs table
  --stage train : RL training (REINFORCE+baseline by default; --ppo for PPO)

Single-GPU only (DDP intentionally not supported here to keep RL simple).

Example:
  CUDA_VISIBLE_DEVICES=0 python train_policy_adn.py --stage cache \
      --weight ./pretrained/yolo-ad-small_0.481_0.453.pt \
      --data bdd100k.yaml --imgsz 1280 --batch 16 \
      --cache-dir ./runs/policy/cache_s_1280

  CUDA_VISIBLE_DEVICES=0 python train_policy_adn.py --stage train \
      --weight ./pretrained/yolo-ad-small_0.481_0.453.pt \
      --data bdd100k.yaml --imgsz 1280 --batch 16 \
      --cache-dir ./runs/policy/cache_s_1280 \
      --out-dir   ./runs/policy/train_s_1280 \
      --epochs 20 --lambda-flops 1.0
"""

import os
import json
import argparse
import math
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import de_parallel


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_anydepth_model(weight_path: str, device: torch.device):
    """Load AnyDepth-YOLO and prepare it for manual forward+loss calls."""
    yolo = YOLO(weight_path, task="detect")
    inner = yolo.model.to(device).eval()  # backbone+head frozen
    n_skip = getattr(inner, "num_skippable_layers", 0)
    assert n_skip > 0, "Loaded model is not an AnyDepth model (num_skippable_layers=0)."
    return yolo, inner, n_skip


def attach_train_args(yolo_model, data_yaml: str, imgsz: int, batch: int, workers: int):
    """
    Build a get_cfg() args namespace and attach to inner model so .loss() works.
    Mirrors what DetectionTrainer.set_model_attributes does.
    """
    overrides = dict(
        task="detect", mode="train", data=data_yaml, imgsz=imgsz,
        batch=batch, workers=workers, rect=True, single_cls=False, cache=False,
    )
    args = get_cfg(overrides=overrides)
    data = check_det_dataset(data_yaml)
    inner = yolo_model.model
    inner.nc = data["nc"]
    inner.names = data["names"]
    inner.args = args
    # Force criterion init (loaded checkpoints may have criterion=None which
    # bypasses the lazy init in DetectionModelAnyDepth.loss()).
    crit, crit_kd = inner.init_criterion()
    inner.criterion = crit
    inner.criterion_kd = crit_kd
    return args, data


def make_dataloader(args, data, split: str, batch: int, shuffle: bool, stride: int):
    """Deterministic (no-augmentation) dataloader for the requested split.
    We use mode='val' regardless of split to disable augmentation — required
    because cached state/L_super/L_base must match what training-time forwards see.
    """
    img_path = data[split]
    dataset = build_yolo_dataset(args, img_path, batch, data, mode="val",
                                 rect=False, stride=stride)
    return build_dataloader(dataset, batch, args.workers, shuffle, rank=-1)


# ---------------------------------------------------------------------------
# FLOPs table (additive estimate using 1+N measurements)
# ---------------------------------------------------------------------------
def measure_flops(inner, x, skip):
    """Forward FLOPs (= 2 * MACs) for a given skip config using thop.
    thop.profile returns MACs by default; multiply by 2 to get FLOPs."""
    try:
        import thop
    except ImportError:
        raise ImportError("Please `pip install thop` for FLOPs measurement.")

    class Wrap(nn.Module):
        def __init__(self, m, sk):
            super().__init__()
            self.m = m
            self.sk = sk
        def forward(self, x):
            return self.m.predict(x, skip=self.sk)
    w = Wrap(inner, list(skip))
    macs, _ = thop.profile(w, inputs=(x,), verbose=False)
    return float(macs) * 2.0  # MACs -> FLOPs


def build_flops_table(inner, n_skip, h, w, device):
    """
    Use additive assumption:
       FLOPs(skip) = FLOPs_base + sum_i (1 - skip[i]) * stage_flops[i]
    Requires (1 + n_skip) measurements only.
    Measured at actual training input shape (h, w) — usually rect-letterboxed.
    """
    inner.eval()
    x = torch.zeros(1, 3, h, w, device=device)
    base = [True] * n_skip
    F_base = measure_flops(inner, x, base)

    stage_flops = []
    for i in range(n_skip):
        sk = [True] * n_skip
        sk[i] = False  # only stage i active
        F_i = measure_flops(inner, x, sk)
        stage_flops.append(F_i - F_base)

    table = {}
    for code in range(2 ** n_skip):
        skip = [(code >> b) & 1 == 1 for b in range(n_skip)]
        f = F_base + sum((0 if skip[i] else stage_flops[i]) for i in range(n_skip))
        table[code] = f
    return table, F_base, stage_flops


def code_to_skip(code: int, n_skip: int):
    return [(code >> b) & 1 == 1 for b in range(n_skip)]


def skip_to_code(skip):
    return sum((1 << b) for b, v in enumerate(skip) if v)


def skip_to_str(skip):
    """[True, False, ...] -> 'TFFFFFFF' (leftmost = stage 0)."""
    return "".join("T" if v else "F" for v in skip)


def str_to_skip(s: str):
    """'TFFFFFFF' -> [True, False, ...] (leftmost = stage 0)."""
    return [c == "T" for c in s]


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
def slice_batch_one(batch, i: int, device):
    """Extract sample i from a collated YOLO batch as a single-image batch."""
    mask = (batch["batch_idx"] == i)
    return {
        "img": batch["img"][i:i+1],
        "batch_idx": torch.zeros(int(mask.sum().item()), device=device,
                                 dtype=batch["batch_idx"].dtype),
        "cls": batch["cls"][mask],
        "bboxes": batch["bboxes"][mask],
    }


# ---------------------------------------------------------------------------
# Stage 1: cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def stage_cache(args_cli):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args_cli.seed)

    yolo, inner, n_skip = load_anydepth_model(args_cli.weight, device)
    train_args, data = attach_train_args(
        yolo, args_cli.data, args_cli.imgsz, args_cli.batch, args_cli.workers
    )
    stride = max(int(de_parallel(inner).stride.max()), 32)
    loader = make_dataloader(train_args, data, split="train", batch=args_cli.batch,
                             shuffle=False, stride=stride)

    cache_dir = Path(args_cli.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ===== Cache config summary =====
    print("=" * 78)
    print("[CACHE STAGE]")
    print(f"  weight       : {args_cli.weight}")
    print(f"  data         : {args_cli.data}")
    print(f"  imgsz        : {args_cli.imgsz}  (rect)")
    print(f"  batch        : {args_cli.batch}")
    print(f"  cache-dir    : {cache_dir}")
    print(f"  max-images   : {args_cli.cache_max_images or 'all'}")
    print(f"  n_skip       : {n_skip}  (action space = 2^{n_skip} = {2**n_skip})")
    print("=" * 78)

    # FLOPs table — use actual rect-letterboxed shape from a real batch
    sample_batch = next(iter(loader))
    _, _, H_in, W_in = sample_batch["img"].shape
    print(f"[*] Measuring FLOPs table at input shape (H={H_in}, W={W_in}) ...")
    t0 = time.time()
    flops_table, F_base, stage_flops = build_flops_table(inner, n_skip, H_in, W_in, device)
    F_super = flops_table[skip_to_code([False] * n_skip)]
    flops_dt = time.time() - t0

    # FLOPs summary
    print(f"    F_base       = {F_base/1e9:7.3f} GFLOPs   (skip = all True)")
    print(f"    F_super      = {F_super/1e9:7.3f} GFLOPs   (skip = all False)")
    print(f"    F_super-base = {(F_super-F_base)/1e9:7.3f} GFLOPs   (max savings range)")
    print(f"    Per-stage FLOPs (when stage is NOT skipped):")
    for i, sf in enumerate(stage_flops):
        print(f"        stage[{i}]  {sf/1e9:7.3f} GFLOPs  ({100*sf/F_super:5.2f}% of SUPER)")
    flop_vals = list(flops_table.values())
    print(f"    All {len(flops_table)} configs: "
          f"min={min(flop_vals)/1e9:.3f}G  max={max(flop_vals)/1e9:.3f}G  "
          f"mean={(sum(flop_vals)/len(flop_vals))/1e9:.3f}G")
    # Print a few representative configs (skip pattern shown as bits, low-bit = stage 0)
    sample_codes = [
        skip_to_code([True]*n_skip),                                  # BASE
        skip_to_code([False]*n_skip),                                 # SUPER
        skip_to_code([True]*(n_skip//2) + [False]*(n_skip - n_skip//2)),
        skip_to_code([False]*(n_skip//2) + [True]*(n_skip - n_skip//2)),
    ]
    print(f"    Representative configs:")
    for c in sample_codes:
        bits = "".join("T" if b else "F" for b in code_to_skip(c, n_skip))
        print(f"        skip={bits}  code={c:3d}  FLOPs={flops_table[c]/1e9:7.3f}G")
    print(f"    [time: {flops_dt:.2f}s]")

    # Save with human-readable string keys ("TTTTTTTT" = BASE, "FFFFFFFF" = SUPER).
    # Leftmost character corresponds to stage 0.
    flops_table_str = {
        skip_to_str(code_to_skip(code, n_skip)): flops_table[code]
        for code in range(2 ** n_skip)
    }
    with open(cache_dir / "flops_table.json", "w") as f:
        json.dump(flops_table_str, f, indent=2, sort_keys=True)
    with open(cache_dir / "flops_meta.json", "w") as f:
        json.dump({
            "unit": "FLOPs (= 2 * MACs)",
            "F_base": F_base,    "skip_BASE":  "T" * n_skip,
            "F_super": F_super,  "skip_SUPER": "F" * n_skip,
            "stage_flops": stage_flops,
            "n_skip": n_skip,
            "input_h": int(H_in),
            "input_w": int(W_in),
            "key_format": "skip pattern string; leftmost char = stage 0; "
                          "T = skip (deeper layer disabled), F = keep",
        }, f, indent=2)
    print(f"[*] Saved flops_table.json ({len(flops_table)} entries) and flops_meta.json")

    # Per-image cache
    SUPER = [False] * n_skip
    BASE = [True] * n_skip

    states = []         # [N, D]
    L_super_all = []    # [N]  per-image scalars
    L_base_all = []     # [N]
    im_files = []       # [N]

    inner.eval()
    n_total = args_cli.cache_max_images or len(loader.dataset)
    print(f"[*] Caching states + L_super + L_base for ~{n_total} images ...")
    t_cache = time.time()
    seen = 0
    for bi, batch in enumerate(loader):
        batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255
        for k in ("batch_idx", "cls", "bboxes"):
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        # SUPER forward (gives features for state)
        out = inner.predict(batch["img"], skip=SUPER, return_features=True)
        feats = out["features"]  # list of [B, C_i] GAP'd vectors
        state = torch.cat(feats, dim=-1)  # [B, sum_C]
        B = state.shape[0]

        # Per-image L_super and L_base (call loss with batch=1 so the returned
        # scalar corresponds to a single image only)
        L_super_b = torch.empty(B)
        L_base_b = torch.empty(B)
        for i in range(B):
            bi_one = slice_batch_one(batch, i, device)
            pred_s = inner.predict(bi_one["img"], skip=SUPER, return_features=False)
            L_s, _ = inner.loss(bi_one, preds=pred_s)
            L_super_b[i] = L_s.detach().cpu()
            pred_b = inner.predict(bi_one["img"], skip=BASE, return_features=False)
            L_b, _ = inner.loss(bi_one, preds=pred_b)
            L_base_b[i] = L_b.detach().cpu()

        states.append(state.detach().cpu())
        L_super_all.extend(L_super_b.tolist())
        L_base_all.extend(L_base_b.tolist())
        im_files.extend(list(batch["im_file"]))

        seen += B
        if bi % 20 == 0:
            elapsed = time.time() - t_cache
            rate = seen / max(elapsed, 1e-6)
            eta = (n_total - seen) / max(rate, 1e-6)
            print(f"    [{seen}/{n_total}] "
                  f"L_super_mean={L_super_b.mean().item():.4f}  "
                  f"L_base_mean={L_base_b.mean().item():.4f}  "
                  f"({rate:.1f} img/s, ETA {eta/60:.1f}m)")

        if args_cli.cache_max_images and seen >= args_cli.cache_max_images:
            break

    states = torch.cat(states, dim=0)
    L_super_t = torch.tensor(L_super_all)
    L_base_t = torch.tensor(L_base_all)
    diff_t = L_base_t - L_super_t

    cache_dt = time.time() - t_cache
    print("=" * 78)
    print(f"[*] Cache loop done in {cache_dt/60:.2f} min "
          f"({states.shape[0]} images, {states.shape[0]/max(cache_dt,1e-6):.1f} img/s)")
    print(f"    state_dim   : {states.shape[1]}")
    print(f"    states size : {states.numel()*4/1e6:.1f} MB (float32)")
    print(f"    L_super     : mean={L_super_t.mean():.4f}  std={L_super_t.std():.4f}  "
          f"min={L_super_t.min():.4f}  max={L_super_t.max():.4f}")
    print(f"    L_base      : mean={L_base_t.mean():.4f}  std={L_base_t.std():.4f}  "
          f"min={L_base_t.min():.4f}  max={L_base_t.max():.4f}")
    print(f"    L_base-L_sup: mean={diff_t.mean():.4f}  std={diff_t.std():.4f}  "
          f"(headroom for policy)")
    n_inverted = int((diff_t < 0).sum().item())
    print(f"    Images where L_base < L_super (inverted): "
          f"{n_inverted}/{len(diff_t)} ({100*n_inverted/len(diff_t):.2f}%)")
    print("=" * 78)

    torch.save({
        "states": states,                         # float32 tensor
        "L_super": L_super_t,                     # per-image scalars
        "L_base": L_base_t,
        "im_files": im_files,
        "state_dim": int(states.shape[1]),
        "n_skip": n_skip,
    }, cache_dir / "cache.pt")
    print(f"[*] Saved cache to {cache_dir / 'cache.pt'}")


# ---------------------------------------------------------------------------
# Policy / Value networks
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    """Factored Bernoulli policy (used at BOTH training & inference).
    Output logits[i] = skip-probability logit for stage i (1 = skip)."""
    def __init__(self, state_dim, n_stages, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_stages)

    def forward(self, s):
        h = F.relu(self.fc1(s))
        h = F.relu(self.fc2(h))
        return self.head(h)  # [B, n_stages]


class ValueNet(nn.Module):
    """State-value baseline. Used ONLY during training (advantage computation).
    Not needed at inference — only PolicyNet is required to deploy the policy."""
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, s):
        h = F.relu(self.fc1(s))
        h = F.relu(self.fc2(h))
        return self.head(h).squeeze(-1)  # [B]


def report_net_size(name: str, net: nn.Module, state_dim: int, device):
    """Print #params and FLOPs (per single state forward) for a small MLP."""
    n_params = sum(p.numel() for p in net.parameters())
    try:
        import thop
        x = torch.zeros(1, state_dim, device=device)
        flops, _ = thop.profile(net, inputs=(x,), verbose=False)
        gflops = flops / 1e9
        flops_str = f"{flops/1e6:.3f} MFLOPs ({gflops:.6f} GFLOPs)"
    except Exception as e:
        flops_str = f"FLOPs unavailable ({e})"
    print(f"[*] {name}: params={n_params:,}  ({n_params/1e3:.2f}K)  {flops_str}")


# ---------------------------------------------------------------------------
# Stage 2: train
# ---------------------------------------------------------------------------
def compute_reward(L_super, L_base, L_pred, F_super, F_base, F_pred,
                   lambda_flops=1.0, scale=1.0,
                   reward_clip=3.0, eps=1e-6):
    """Lagrangian reward:  r = quality - lambda * flops_extra.

    quality      = (L_super - L_pred) / scale
                   ~= 0  when L_pred matches SUPER  (no quality loss)
                   ~= -1 when L_pred matches BASE   (because scale = mean ΔL)
                   > 0  when L_pred beats SUPER     (rare but valid; can happen
                                                     on "inverted" samples where
                                                     BASE > SUPER quality-wise)
    flops_extra  = (F_pred - F_base) / (F_super - F_base)   in [0, 1]
                   0 at BASE, 1 at SUPER.

    `scale` is a single dataset-level constant = mean(L_base - L_super) over the
    training cache, passed in by stage_train. Using a constant (not per-sample
    |L_base - L_super|) avoids amplifying noise on easy/inverted images and
    removes the need for a `delta_min` floor.

    Reference rewards (with lambda=1.0, on average):
        BASE  (skip all):   -1 - 0     = -1
        SUPER (no skip):     0 - 1     = -1     <- equal!
        Inverted sample, BASE: +(L_super-L_base)/scale - 0 = positive  <- correctly preferred
        Hard sample, SUPER:    0 - 1            <- beats BASE which scores < -1

    Setting BASE and SUPER to the same average reward forces the policy to
    *discriminate per image* to gain reward — there is no free lunch from
    always-skip or always-keep.

    Tuning lambda:
      lambda < 1: bias toward SUPER (accuracy-preferring)
      lambda = 1: balanced (recommended starting point)
      lambda > 1: bias toward BASE (efficiency-preferring)
    """
    quality     = (L_super - L_pred) / max(scale, eps)
    flops_extra = (F_pred - F_base) / (F_super - F_base + eps)
    reward = quality - lambda_flops * flops_extra
    reward = reward.clamp(-reward_clip, reward_clip)
    # return quality (acc-like; higher=better) and 1-flops_extra (effi-like) for logging
    return reward, quality, 1.0 - flops_extra


def stage_train(args_cli):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args_cli.seed)
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args_cli.cache_dir)
    cache = torch.load(cache_dir / "cache.pt", map_location="cpu")
    flops_meta = json.load(open(cache_dir / "flops_meta.json"))
    F_base = flops_meta["F_base"]
    F_super = flops_meta["F_super"]
    n_skip = flops_meta["n_skip"]
    # Convert string keys ("TTTTTTTT") -> integer codes for fast indexing.
    flops_table_str = json.load(open(cache_dir / "flops_table.json"))
    flops_table = {skip_to_code(str_to_skip(k)): v
                   for k, v in flops_table_str.items()}

    state_dim = cache["state_dim"]

    # Reward calibration: mean(L_base - L_super) on the cached training set.
    # Used as the `scale` constant in the Lagrangian reward so that picking BASE
    # yields quality ~= -1 on average, matching SUPER's penalty of -lambda when
    # lambda=1.
    diff_t = (cache["L_base"] - cache["L_super"]).float()
    reward_scale = float(diff_t.mean().item())
    n_inverted = int((diff_t < 0).sum().item())

    # ===== Train config summary =====
    print("=" * 78)
    print("[TRAIN STAGE]")
    print(f"  weight       : {args_cli.weight}")
    print(f"  data         : {args_cli.data}")
    print(f"  imgsz/batch  : {args_cli.imgsz} / {args_cli.batch}")
    print(f"  out-dir      : {out_dir}")
    print(f"  cache-dir    : {cache_dir}")
    print(f"  cache size   : N={len(cache['im_files'])} state_dim={state_dim}")
    print(f"  algorithm    : {'PPO (clip)' if args_cli.ppo else 'REINFORCE+baseline'}"
          f"   adv_norm={args_cli.adv_norm}")
    print(f"  hyperparams  : lr={args_cli.lr}  hidden={args_cli.hidden}  "
          f"epochs={args_cli.epochs}")
    print(f"                 lambda_flops={args_cli.lambda_flops}  "
          f"c_v={args_cli.c_v}  c_e={args_cli.c_e}"
          + (f"  clip={args_cli.clip}" if args_cli.ppo else ""))
    print(f"  reward       : Lagrangian  r = quality - lambda * flops_extra")
    print(f"                 quality = (L_super - L_pred) / scale")
    print(f"                 flops_extra = (F_pred - F_base) / (F_super - F_base)")
    print(f"                 scale = mean(L_base - L_super) = {reward_scale:.4f}  "
          f"(from cache; {n_inverted}/{len(diff_t)} inverted samples)")
    print(f"                 reward_clip=±{args_cli.reward_clip}")
    print(f"  FLOPs        : F_base={F_base/1e9:.3f}G  F_super={F_super/1e9:.3f}G  "
          f"range={(F_super-F_base)/1e9:.3f}G")
    print(f"  n_skip       : {n_skip}  (action space = 2^{n_skip} = {2**n_skip})")
    print("=" * 78)
    if reward_scale <= 0:
        print(f"[!] WARNING: reward_scale = {reward_scale:.4f} <= 0. This means BASE "
              f"is, on average, BETTER than SUPER on the cached set — the underlying "
              f"detector is not benefiting from the deeper layers. Policy training "
              f"will collapse to BASE regardless of reward design. Please retrain "
              f"the detector with proper super-stage utilization before running RL.")
        return

    # Detection model (frozen) for L_pred forward
    yolo, inner, _ = load_anydepth_model(args_cli.weight, device)
    train_args, data = attach_train_args(
        yolo, args_cli.data, args_cli.imgsz, args_cli.batch, args_cli.workers
    )
    for p in inner.parameters():
        p.requires_grad_(False)
    inner.eval()
    stride = max(int(de_parallel(inner).stride.max()), 32)
    loader = make_dataloader(train_args, data, split="train", batch=args_cli.batch,
                             shuffle=True, stride=stride)

    # Map im_file -> cache idx for lookup during training
    file2idx = {f: i for i, f in enumerate(cache["im_files"])}

    # Policy / Value networks
    policy = PolicyNet(state_dim, n_skip, hidden=args_cli.hidden).to(device)
    value = ValueNet(state_dim, hidden=args_cli.hidden).to(device)
    report_net_size("PolicyNet (used at train & inference)", policy, state_dim, device)
    report_net_size("ValueNet  (training-only baseline)   ", value, state_dim, device)
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=args_cli.lr,
    )

    # FLOPs lookup tensor for fast indexing
    flops_arr = torch.zeros(2 ** n_skip)
    for k, v in flops_table.items():
        flops_arr[k] = v
    flops_arr = flops_arr.to(device)

    # Keep big tensors on CPU; transfer per-batch to save GPU memory.
    cache_states = cache["states"]                # CPU float32 [N, D]
    cache_L_super = cache["L_super"]              # CPU float32 [N]
    cache_L_base = cache["L_base"]                # CPU float32 [N]

    # Training loop
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")
    step = 0
    t_train = time.time()
    for epoch in range(args_cli.epochs):
        # Per-epoch accumulators
        ep_reward, ep_acc, ep_effi, ep_Fpred = [], [], [], []
        ep_neg = 0
        ep_total = 0
        ep_per_stage_skip = torch.zeros(n_skip)
        ep_action_codes = Counter()
        t_epoch = time.time()
        for bi, batch in enumerate(loader):
            batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255
            for k in ("batch_idx", "cls", "bboxes"):
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)

            # Lookup cached states & losses for this batch
            try:
                idxs = torch.tensor([file2idx[f] for f in batch["im_file"]],
                                    dtype=torch.long, device=device)
            except KeyError as e:
                # image not in cache (e.g., cache_max_images limited) -> skip
                continue

            idxs_cpu = idxs.cpu()
            s = cache_states[idxs_cpu].to(device, non_blocking=True)        # [B, D]
            L_super_b = cache_L_super[idxs_cpu].to(device, non_blocking=True)
            L_base_b = cache_L_base[idxs_cpu].to(device, non_blocking=True)
            B = s.shape[0]

            # --- Sample action from policy ---
            logits = policy(s)
            dist = Bernoulli(logits=logits)
            action = dist.sample()                          # [B, n_skip] (1=skip)
            logp = dist.log_prob(action).sum(-1)            # [B]
            entropy = dist.entropy().sum(-1).mean()

            # --- Forward with sampled action (no grad through detection) ---
            with torch.no_grad():
                # Each sample in batch can have different skip; AnyDepth predict
                # accepts a single skip list per forward, so loop per-sample.
                # (For BDD100K batch sizes this is acceptable; fully vectorized
                # would require model-level support.)
                L_pred = torch.empty(B, device=device)
                F_pred = torch.empty(B, device=device)
                for i in range(B):
                    skip_i = [bool(v) for v in action[i].tolist()]
                    code = skip_to_code(skip_i)
                    F_pred[i] = flops_arr[code]
                    bi_one = slice_batch_one(batch, i, device)
                    pred = inner.predict(bi_one["img"], skip=skip_i)
                    L_i, _ = inner.loss(bi_one, preds=pred)
                    L_pred[i] = L_i.detach()

            # --- Reward ---
            reward, acc, effi = compute_reward(
                L_super_b, L_base_b, L_pred,
                F_super, F_base, F_pred,
                lambda_flops=args_cli.lambda_flops,
                scale=reward_scale,
                reward_clip=args_cli.reward_clip,
            )

            # --- Advantage with value baseline ---
            v = value(s)
            adv = (reward - v.detach())
            if args_cli.adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # --- Losses ---
            if args_cli.ppo:
                # Need old logp; here we use single-step PPO with K=1, so ratio=1.
                # For full PPO with K>1 epochs over same rollout, store old logp
                # and re-sample. Keeping it minimal here.
                ratio = torch.exp(logp - logp.detach())
                clipped = torch.clamp(ratio, 1 - args_cli.clip, 1 + args_cli.clip)
                L_policy = -torch.min(ratio * adv, clipped * adv).mean()
            else:
                L_policy = -(logp * adv).mean()                  # REINFORCE w/ baseline
            L_value = F.mse_loss(v, reward)
            L_ent = -entropy
            loss = L_policy + args_cli.c_v * L_value + args_cli.c_e * L_ent

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value.parameters()), 1.0
            )
            optim.step()

            # --- Logging ---
            with torch.no_grad():
                mean_skip = action.float().mean().item()
                fpred_mean = F_pred.mean().item() / 1e9
                neg_ratio_step = float((acc < 0).float().mean().item())
                rec = {
                    "epoch": epoch, "step": step,
                    "reward": reward.mean().item(),
                    "acc": acc.mean().item(),
                    "effi": effi.mean().item(),
                    "F_pred_G": fpred_mean,
                    "L_pred": L_pred.mean().item(),
                    "policy_loss": float(L_policy.item()),
                    "value_loss": float(L_value.item()),
                    "entropy": float(entropy.item()),
                    "mean_skip_rate": mean_skip,
                    "neg_acc_ratio": neg_ratio_step,
                }

                # Accumulate epoch stats
                ep_reward.append(reward.mean().item())
                ep_acc.append(acc.mean().item())
                ep_effi.append(effi.mean().item())
                ep_Fpred.append(F_pred.mean().item())
                ep_neg += int((acc < 0).sum().item())
                ep_total += B
                ep_per_stage_skip += action.float().sum(dim=0).cpu()
                for i in range(B):
                    ep_action_codes[skip_to_code(action[i].tolist())] += 1

            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if step % 10 == 0:
                print(f"[ep{epoch} st{step}] r={rec['reward']:+.3f} "
                      f"acc={rec['acc']:+.3f} effi={rec['effi']:.3f} "
                      f"F={rec['F_pred_G']:.2f}G "
                      f"H={rec['entropy']:.3f} "
                      f"skip%={mean_skip:.2f} neg%={neg_ratio_step:.2f}")
            step += 1

        # ===== Per-epoch summary =====
        ep_dt = time.time() - t_epoch
        per_stage_rate = (ep_per_stage_skip / max(ep_total, 1)).tolist()
        top5 = ep_action_codes.most_common(5)
        print("-" * 78)
        print(f"[EPOCH {epoch} SUMMARY]  ({ep_dt/60:.1f} min, {ep_total} samples)")
        print(f"  reward      : mean={np.mean(ep_reward):+.4f}  "
              f"quality={np.mean(ep_acc):+.4f}  effi={np.mean(ep_effi):.4f}")
        print(f"  F_pred mean : {np.mean(ep_Fpred)/1e9:.3f} GFLOPs  "
              f"({100*(F_super-np.mean(ep_Fpred))/(F_super-F_base):.1f}% savings vs SUPER)")
        print(f"  worse-than-SUPER (quality<0): {ep_neg}/{ep_total} "
              f"({100*ep_neg/max(ep_total,1):.2f}%)")
        print(f"  per-stage skip rate (1=skip):")
        print(f"    " + "  ".join(f"s{i}={r:.2f}" for i, r in enumerate(per_stage_rate)))
        print(f"  top-5 chosen configs (skip pattern, low-bit=stage 0):")
        for code, cnt in top5:
            bits = "".join("T" if b else "F" for b in code_to_skip(code, n_skip))
            f_pct = 100 * (F_super - flops_arr[code].item()) / (F_super - F_base)
            print(f"    {bits}  code={code:3d}  count={cnt:5d} "
                  f"({100*cnt/max(ep_total,1):5.2f}%)  "
                  f"F={flops_arr[code].item()/1e9:.2f}G  ({f_pct:+.1f}% savings)")
        print("-" * 78)

        # Save checkpoint per epoch
        ckpt = {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "epoch": epoch,
            "args": vars(args_cli),
            "state_dim": state_dim,
            "n_skip": n_skip,
        }
        torch.save(ckpt, out_dir / f"policy_ep{epoch:03d}.pt")
        torch.save(ckpt, out_dir / "policy_last.pt")
        print(f"[*] Saved checkpoint epoch {epoch} -> {out_dir/'policy_last.pt'}")

        # Append per-epoch summary to a separate JSONL for easy plotting
        with open(out_dir / "epoch_summary.jsonl", "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "minutes": ep_dt / 60,
                "reward_mean": float(np.mean(ep_reward)),
                "acc_mean": float(np.mean(ep_acc)),
                "effi_mean": float(np.mean(ep_effi)),
                "F_pred_mean_G": float(np.mean(ep_Fpred)) / 1e9,
                "F_savings_pct": float(100 * (F_super - np.mean(ep_Fpred)) /
                                       (F_super - F_base)),
                "neg_acc_ratio": ep_neg / max(ep_total, 1),
                "per_stage_skip_rate": per_stage_rate,
                "top5_codes": [(c, n) for c, n in top5],
            }) + "\n")

    print("=" * 78)
    print(f"[TRAIN DONE] total {(time.time()-t_train)/60:.1f} min, "
          f"{step} steps over {args_cli.epochs} epochs")
    print(f"[*] PolicyNet weights: {out_dir/'policy_last.pt'}")
    print("=" * 78)
    log_f.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["cache", "train"], required=True)
    p.add_argument("--weight", required=True, help="AnyDepth-YOLO checkpoint (.pt)")
    p.add_argument("--data", default="bdd100k.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--cache-max-images", type=int, default=0,
                   help="0 = use full train set")

    # train-only
    p.add_argument("--out-dir", default="./runs/policy/train")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lambda-flops", type=float, default=1.0, dest="lambda_flops",
                   help="Lagrangian weight on FLOPs penalty. "
                        "lambda<1: accuracy-preferring; lambda=1: balanced "
                        "(SUPER and BASE earn equal avg reward); lambda>1: "
                        "efficiency-preferring.")
    p.add_argument("--c_v", type=float, default=0.5)
    p.add_argument("--c_e", type=float, default=0.05,
                   help="Entropy bonus coefficient (raise to delay collapse).")
    p.add_argument("--reward-clip", type=float, default=3.0,
                   help="Symmetric clip on the final reward to bound RL variance.")
    p.add_argument("--no-adv-norm", dest="adv_norm", action="store_false",
                   help="Disable per-batch advantage normalization (default: enabled)")
    p.set_defaults(adv_norm=True)
    p.add_argument("--ppo", action="store_true",
                   help="Use PPO clip (default: REINFORCE+baseline)")
    p.add_argument("--clip", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "cache":
        stage_cache(args)
    elif args.stage == "train":
        stage_train(args)


if __name__ == "__main__":
    main()
