"""On-device latency-budget tracking on Jetson Orin Nano (Fig.8, live closed loop).

This is the budget-tracking half of the scenario figure, stripped of the
domain-shift framing (night<->day / rain), which needs labelled video we do not
have. What remains is the part that validates the controller: given a time-varying
target latency budget L*(t), does the realized mean latency track it on THIS device?

Fully live, no replay, no fluke:
  * Real (unlabelled) KITTI-Tracking *testing* frames stream through the detector.
    Budget tracking measures only LATENCY, never accuracy, so labels are irrelevant.
  * Each frame runs EXACTLY ONE depth path (BASE or SUPER) on the real detector;
    its tap features feed the router, which predicts the advantage Ahat ON-DEVICE;
    a PI controller adjusts the threshold tau so the MEASURED realized latency
    tracks L*(t). Everything (detector + router + control) executes on the Orin Nano.

Timed scope per frame = detector inference + router (preprocess and NMS excluded),
matching the TRT-deployment scope in the paper. The detector head still runs inside
the engine; we simply skip NMS post-processing (not needed to time the loop).

Two backends (pick whichever asset you built/copied on the Jetson):
  * TRT  : --base base.fp16.engine --super super.fp16.engine  (built on THIS device,
           from export_onnx.py which emits raw tap maps feat{4,6,8,14,17,20})
  * Torch: --weight anydepth.pt                                (AnyDepth .pt, skip-routed)

Usage (TRT):
    python trt_bench/jetson_budget_track.py \
        --base  trt_bench/onnx/kitti/base.fp16.engine \
        --super trt_bench/onnx/kitti/super.fp16.engine \
        --policy method02_advantage_regress_tinyConv/outputs/kitti/ablation/policy_both_g2_s0.pt \
        --kitti_root /media/data/kitti-tracking --imgsz 384 1248 \
        --out trt_bench/jetson_budget.pdf

Usage (PyTorch):
    python trt_bench/jetson_budget_track.py \
        --weight finetuned_kitti/best.pt \
        --policy method02_advantage_regress_tinyConv/outputs/kitti/ablation/policy_both_g2_s0.pt \
        --kitti_root /media/data/kitti-tracking --imgsz 384 1248 \
        --out trt_bench/jetson_budget.pdf
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox


# --------------------------------------------------------------------------- #
# Jetson on-board power (INA3221 via hwmon sysfs). NVML/pynvml does NOT report
# power on Tegra, so we read the rails directly: power_mW = volt_mV * curr_mA / 1000.
#   VDD_IN          = total module power (what the wall sees)
#   VDD_CPU_GPU_CV  = compute power (CPU + GPU + CV engines)
# --------------------------------------------------------------------------- #
class JetsonPower:
    RAILS = ("VDD_IN", "VDD_CPU_GPU_CV")

    def __init__(self):
        self.chans = {}                      # rail -> (volt_path, curr_path)
        for label_f in glob.glob("/sys/class/hwmon/hwmon*/in*_label"):
            try:
                lbl = open(label_f).read().strip()
            except OSError:
                continue
            if lbl not in self.RAILS:
                continue
            m = re.search(r"in(\d+)_label$", label_f)
            if not m:
                continue
            d = Path(label_f).parent
            ch = m.group(1)
            v, c = d / f"in{ch}_input", d / f"curr{ch}_input"
            if v.exists() and c.exists():
                self.chans[lbl] = (str(v), str(c))
        self.available = bool(self.chans)
        if not self.available:
            print("[!] no INA3221 power rails found; energy = 0")

    def power_mw(self, rail="VDD_IN"):
        """Instantaneous power on a rail in mW (volt_mV * curr_mA / 1000)."""
        ch = self.chans.get(rail)
        if ch is None:
            return 0.0
        try:
            return float(open(ch[0]).read()) * float(open(ch[1]).read()) / 1000.0
        except (OSError, ValueError):
            return 0.0

from method02_advantage_regress_tinyConv.feature_tap import (
    INPUT_LEVEL_LAYERS, PRED_LEVEL_LAYERS, STATE_LAYERS)
from method02_advantage_regress_tinyConv.eval_video_bdd import load_policy, grid_vec


# --------------------------------------------------------------------------- #
# Backends. Each exposes run_frame(x, choice) -> (ahat, measured_ms): execute one
# depth path on a preprocessed frame, run the router on its taps, CUDA-event time
# the (detector inference + router) scope. Both keep the router resident.
# --------------------------------------------------------------------------- #
class TRTBackend:
    def __init__(self, base_path, super_path, policy, grid, device, router_engine=None):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.dev = device
        self.grid = grid
        self.eng = {"base": self._load(base_path), "super": self._load(super_path)}
        self.net, self.ckpt, self.feat, self.is_gap = load_policy(policy, device)
        self.use_pred = self.feat in ("both", "pred")
        self.pid = {"base": torch.zeros(1, dtype=torch.long, device=device),
                    "super": torch.ones(1, dtype=torch.long, device=device)}
        # Whether the detector engines already emit pre-pooled (C,2,2) taps (pooled
        # export) -- then the host pooling below is a trivial 2x2->2x2 identity.
        self.pooled_taps = self.eng["base"]["out"]["feat4"].shape[-1] == grid
        # Optional single TRT router engine (iv, pv) -> logit. path_id is a training
        # technique, so deployment uses one fixed router engine for all frames.
        self.re = self._load_router(router_engine) if router_engine else None
        if self.re is None:
            # Materialize the lazy eager router with real tap dims, then load weights.
            in_c = sum(self.eng["base"]["out"][f"feat{l}"].shape[1] for l in INPUT_LEVEL_LAYERS)
            pr_c = sum(self.eng["base"]["out"][f"feat{l}"].shape[1] for l in PRED_LEVEL_LAYERS)
            with torch.no_grad():
                self.net(torch.zeros(2, in_c, grid, grid, device=device),
                         torch.zeros(2, pr_c, grid, grid, device=device) if self.use_pred else None,
                         torch.zeros(2, dtype=torch.long, device=device))
            self.net.load_state_dict(self.ckpt["state_dict"]); self.net.eval()
        self.stream = torch.cuda.Stream()
        self.ev0 = torch.cuda.Event(enable_timing=True)
        self.ev1 = torch.cuda.Event(enable_timing=True)

    def _load(self, path):
        with open(path, "rb") as f:
            engine = self.trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()
        inp, out = None, {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dt = self.trt.nptype(engine.get_tensor_dtype(name))
            t = torch.zeros(shape, dtype=getattr(torch, np.dtype(dt).name), device=self.dev)
            ctx.set_tensor_address(name, t.data_ptr())
            if engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                inp = t
            else:
                out[name] = t
        return {"ctx": ctx, "inp": inp, "out": out}

    def _load_router(self, path):
        """Load the (iv, pv) -> logit router engine, keeping inputs/output by name."""
        with open(path, "rb") as f:
            engine = self.trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()
        io = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dt = self.trt.nptype(engine.get_tensor_dtype(name))
            t = torch.zeros(tuple(engine.get_tensor_shape(name)),
                            dtype=getattr(torch, np.dtype(dt).name), device=self.dev)
            ctx.set_tensor_address(name, t.data_ptr())
            io[name] = t
        return {"ctx": ctx, "io": io}

    def _pool_taps(self, out, layers):
        """Cat the tap maps for `layers` into [1, sumC, grid, grid]. With pooled-tap
        engines this is a 2x2->2x2 identity (near-free); otherwise it pools on host."""
        if self.pooled_taps:
            return torch.cat([out[f"feat{l}"] for l in layers], dim=1)
        return torch.cat([F.adaptive_avg_pool2d(out[f"feat{l}"].float(), self.grid).squeeze(0)
                          for l in layers], dim=0).unsqueeze(0)

    @torch.no_grad()
    def run_frame(self, x, choice):
        e = self.eng[choice]
        with torch.cuda.stream(self.stream):
            e["inp"].copy_(x)
            self.ev0.record(self.stream)
            e["ctx"].execute_async_v3(self.stream.cuda_stream)
            iv = self._pool_taps(e["out"], INPUT_LEVEL_LAYERS)
            pv = self._pool_taps(e["out"], PRED_LEVEL_LAYERS) if self.use_pred else None
            if self.re is not None:
                self.re["io"]["iv"].copy_(iv.to(self.re["io"]["iv"].dtype))
                if pv is not None:
                    self.re["io"]["pv"].copy_(pv.to(self.re["io"]["pv"].dtype))
                self.re["ctx"].execute_async_v3(self.stream.cuda_stream)
                ahat = self.re["io"]["logit"].view(-1)
            else:
                if self.is_gap:
                    iv = iv.mean(dim=(2, 3)); pv = pv.mean(dim=(2, 3)) if pv is not None else None
                ahat = self.net.logit(iv, pv, self.pid[choice]).view(-1)
            self.ev1.record(self.stream)
        self.ev1.synchronize()
        return float(ahat), self.ev0.elapsed_time(self.ev1)


class TorchBackend:
    def __init__(self, weight, policy, grid, imgsz, device):
        from ultralytics import YOLO
        self.dev = device
        self.grid = grid
        Hd = (imgsz[0] + 31) // 32 * 32
        Wd = (imgsz[1] + 31) // 32 * 32
        self.yolo = YOLO(weight, task="detect")
        self.model = self.yolo.model.to(device).eval()
        N = getattr(self.model, "num_skippable_layers", 0)
        if N <= 0:
            raise SystemExit("AnyDepth model required (num_skippable_layers == 0).")
        self.skip = {"base": [True] * N, "super": [False] * N}
        self.cap = {}
        for l in STATE_LAYERS:
            self.model.model[l].register_forward_hook(
                lambda m, i, o, k=l: self.cap.__setitem__(k, o))
        self.net, self.ckpt, self.feat, self.is_gap = load_policy(policy, device)
        self.use_pred = self.feat in ("both", "pred")
        self.pid = {"base": torch.zeros(1, dtype=torch.long, device=device),
                    "super": torch.ones(1, dtype=torch.long, device=device)}
        # one dummy forward to size the lazy router, then load weights
        with torch.no_grad():
            self.model._predict_once(torch.zeros(1, 3, Hd, Wd, device=device),
                                     skip=self.skip["base"], return_features=False)
            iv = grid_vec(self.cap, INPUT_LEVEL_LAYERS, grid).unsqueeze(0)
            pv = grid_vec(self.cap, PRED_LEVEL_LAYERS, grid).unsqueeze(0) if self.use_pred else None
            if self.is_gap:
                iv = iv.mean(dim=(2, 3)); pv = pv.mean(dim=(2, 3)) if pv is not None else None
            self.net(iv.repeat(2, *([1] * (iv.dim() - 1))),
                     pv.repeat(2, *([1] * (pv.dim() - 1))) if pv is not None else None,
                     torch.zeros(2, dtype=torch.long, device=device))
        self.net.load_state_dict(self.ckpt["state_dict"]); self.net.eval()
        self.ev0 = torch.cuda.Event(enable_timing=True)
        self.ev1 = torch.cuda.Event(enable_timing=True)

    @torch.no_grad()
    def run_frame(self, x, choice):
        self.cap.clear()
        self.ev0.record()
        self.model._predict_once(x, skip=self.skip[choice], return_features=False)
        iv = grid_vec(self.cap, INPUT_LEVEL_LAYERS, self.grid).unsqueeze(0)
        pv = grid_vec(self.cap, PRED_LEVEL_LAYERS, self.grid).unsqueeze(0) if self.use_pred else None
        if self.is_gap:
            iv = iv.mean(dim=(2, 3)); pv = pv.mean(dim=(2, 3)) if pv is not None else None
        ahat = self.net.logit(iv, pv, self.pid[choice]).view(-1)
        self.ev1.record()
        self.ev1.synchronize()
        return float(ahat), self.ev0.elapsed_time(self.ev1)


# --------------------------------------------------------------------------- #
# Target budget schedules (between the live-measured base/super anchors).
# --------------------------------------------------------------------------- #
def target_schedule(kind, n, lo, hi):
    t = np.arange(n)
    if kind == "step":
        seg = np.array([0.30, 0.80, 0.50])
        edges = np.linspace(0, n, len(seg) + 1).astype(int)
        L = np.empty(n)
        for i, frac in enumerate(seg):
            L[edges[i]:edges[i + 1]] = lo + frac * (hi - lo)
        return L
    if kind == "sawtooth":
        period = max(1, n // 4)
        return lo + ((t % period) / period) * (hi - lo)
    raise ValueError(kind)


def trail(x, w):
    """Zero-phase (centered) moving average for visualization only (no phase lag);
    the closed loop itself is unchanged."""
    x = np.asarray(x, float)
    h = w // 2
    xp = np.pad(x, (h, w - 1 - h), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


# --------------------------------------------------------------------------- #
# Live closed-loop PI controller (ms-scale error; ported from online_budget_demo).
# --------------------------------------------------------------------------- #
def online_loop(backend, frames, loader, Ltgt, l_base, tau_hi, tau_lo, kp, ki, beta,
                power=None):
    n = len(frames)
    config = "base"
    tau0 = 0.5 * (tau_hi + tau_lo)        # actuator reference (center of the tau band)
    tau = tau_hi
    integ = 0.0
    L_ema = l_base
    realized = np.empty(n)
    is_super = np.zeros(n, dtype=bool)    # path actually run on each frame
    switched = np.zeros(n, dtype=bool)    # this frame's engine differs from previous
    energy_in = np.zeros(n)               # per-frame energy on VDD_IN (mJ)
    energy_cgc = np.zeros(n)              # per-frame energy on VDD_CPU_GPU_CV (mJ)
    n_super = 0
    prev_config = None
    for t, (path, first) in enumerate(frames):
        if first:
            config = "base"           # each sequence starts on the base path
        x = loader(path)              # decode+preprocess one frame (only one on GPU)
        # Sample power around the (CUDA-timed) compute; energy = P_avg * latency.
        pin0 = power.power_mw("VDD_IN") if power else 0.0
        pcg0 = power.power_mw("VDD_CPU_GPU_CV") if power else 0.0
        ahat, lat = backend.run_frame(x, config)
        pin1 = power.power_mw("VDD_IN") if power else 0.0
        pcg1 = power.power_mw("VDD_CPU_GPU_CV") if power else 0.0
        realized[t] = lat
        is_super[t] = (config == "super")
        switched[t] = (prev_config is not None and config != prev_config)
        prev_config = config
        energy_in[t] = 0.5 * (pin0 + pin1) * lat / 1000.0
        energy_cgc[t] = 0.5 * (pcg0 + pcg1) * lat / 1000.0
        n_super += (config == "super")
        L_ema = beta * L_ema + (1 - beta) * lat
        e = Ltgt[t] - L_ema
        # Positional PI: tau = tau0 - (Kp*e + Ki*integ). Raising tau routes fewer
        # frames to SUPER -> lower latency, so the actuator moves OPPOSITE the error
        # (realized too low, e>0 -> lower tau -> more SUPER). Back-calculation
        # anti-windup: after clipping tau to its rails, pull the integrator back so
        # it stays consistent with the applied tau (integ += (tau_un - tau_cl)/ki),
        # which un-saturates the instant the error reverses -- without this the
        # integral winds up and gets stuck at a rail on a falling target.
        integ += e
        tau_un = tau0 - kp * e - ki * integ
        tau = float(np.clip(tau_un, tau_lo, tau_hi))
        if ki:
            integ += (tau_un - tau) / ki
        config = "super" if ahat > tau else "base"
    return realized, n_super / n, energy_in, energy_cgc, is_super, switched


def render(dump, win, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif", "mathtext.fontset": "stix"})
    l_base, l_super = dump["l_base"], dump["l_super"]
    budgets = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]
    ylo, yhi = l_base - 1.2, l_super + 1.6
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), squeeze=False)
    print(f"\n{'budget':<10}{'MAE(ms)':>9}{'SUPER%':>9}")
    for c, (bkind, btitle) in enumerate(budgets):
        cell = dump["cells"][bkind]
        realized = np.asarray(cell["realized"])
        Ltgt = np.asarray(cell["target"])
        n = len(realized)
        sm = trail(realized, win)
        mae = float(np.mean(np.abs(sm - Ltgt)))
        print(f"{bkind:<10}{mae:>9.3f}{cell['super_rate'] * 100:>8.1f}%")
        ax = axes[0][c]
        ax.plot(Ltgt, color="black", ls="--", lw=1.3, zorder=6)
        ax.plot(sm, color="tab:red", lw=1.5, zorder=5)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0, n)
        ax.grid(alpha=0.2, ls="--")
        ax.tick_params(labelsize=7)
        ax.set_title(btitle, fontsize=9)
        ax.set_xlabel("frame", fontsize=8)
        if c == 0:
            ax.set_ylabel("latency (ms)", fontsize=8)
        txt = f"MAE={mae:.2f} ms"
        if cell.get("mean_mj_in"):
            txt += f"\n{cell['mean_mj_in']:.0f} mJ/frame ({cell.get('mean_W_in', 0):.1f} W)"
        ax.text(0.02, 0.04, txt, transform=ax.transAxes,
                va="bottom", ha="left", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    handles = [Line2D([0], [0], color="black", ls="--", lw=1.3, label="target budget $L^\\star(t)$"),
               Line2D([0], [0], color="tab:red", lw=1.5, label="realized mean latency")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    dev = dump.get("device_name", "Jetson Orin Nano")
    fig.suptitle(f"Runtime latency-budget tracking on {dev} "
                 f"(BASE={l_base:.1f}ms / SUPER={l_super:.1f}ms)", fontsize=9.5)
    fig.tight_layout(pad=0.4, rect=(0, 0.04, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="BASE TRT engine (.engine)")
    ap.add_argument("--super", default=None, help="SUPER TRT engine (.engine)")
    ap.add_argument("--weight", default=None, help="AnyDepth .pt (PyTorch backend)")
    ap.add_argument("--policy", required=True, help="router checkpoint (.pt)")
    ap.add_argument("--router_engine", default=None,
                    help="optional single TRT router engine (iv,pv)->logit; if set, the "
                         "router runs in TRT instead of eager PyTorch")
    ap.add_argument("--kitti_root", default="/media/data/kitti-tracking")
    ap.add_argument("--split", default="testing", help="testing (unlabelled) or training")
    ap.add_argument("--sequences", type=str, nargs="*", default=None, help="seq subset")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1248], help="H W")
    ap.add_argument("--grid", type=int, default=2, help="router feature grid (match policy)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--kp", type=float, default=0.040)
    ap.add_argument("--ki", type=float, default=0.020)
    ap.add_argument("--beta", type=float, default=0.85)
    ap.add_argument("--win", type=int, default=60, help="display smoothing window")
    ap.add_argument("--limit", type=int, default=0, help="cap total frames (debug)")
    ap.add_argument("--out", default="trt_bench/jetson_budget.pdf")
    ap.add_argument("--dump_out", default="trt_bench/jetson_budget.json")
    ap.add_argument("--replot", action="store_true", help="re-render from saved json (no inference)")
    args = ap.parse_args()

    if args.replot:
        render(json.loads(Path(args.dump_out).read_text()), args.win, args.out)
        return

    dev = args.device if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("CUDA required for on-device latency measurement.")

    if args.base and args.super:
        print(f"[*] TRT backend: {args.base} / {args.super}")
        backend = TRTBackend(args.base, args.super, args.policy, args.grid, dev,
                             router_engine=args.router_engine)
    elif args.weight:
        print(f"[*] PyTorch backend: {args.weight}")
        backend = TorchBackend(args.weight, args.policy, args.grid, args.imgsz, dev)
    else:
        raise SystemExit("Provide either --base/--super engines or --weight.")
    print(f"[*] router: {args.policy}  (feat={backend.feat}, gap={backend.is_gap})")

    try:
        import pynvml
        pynvml.nvmlInit()
        dev_name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        dev_name = dev_name.decode() if isinstance(dev_name, bytes) else dev_name
    except Exception:
        dev_name = torch.cuda.get_device_name(0)
    print(f"[*] device: {dev_name}")
    power = JetsonPower()
    if power.available:
        print(f"[*] power rails: {', '.join(power.chans)}")

    # ---- load the (unlabelled) testing stream: concatenate sequences ----
    H = (args.imgsz[0] + 31) // 32 * 32
    W = (args.imgsz[1] + 31) // 32 * 32
    lb = LetterBox((H, W), auto=False, stride=32)

    def preprocess(bgr):
        img = lb(image=bgr)
        t = torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()).to(dev)
        return (t.float() / 255.0).unsqueeze(0)

    def loader(path):
        """Decode + letterbox one frame to a GPU tensor (only one resident at a time;
        this is the preprocess step, kept OUTSIDE the per-frame CUDA-event timing)."""
        bgr = cv2.imread(path)
        return preprocess(bgr) if bgr is not None else None

    img_root = Path(args.kitti_root) / args.split / "image_02"
    seqs = args.sequences or sorted(p.name for p in img_root.iterdir() if p.is_dir())
    frames = []          # list of (path, first_of_seq); decoded lazily in the loop
    for seq in seqs:
        fps = sorted((img_root / seq).glob("*.png")) or sorted((img_root / seq).glob("*.jpg"))
        first = True
        for p in fps:
            frames.append((str(p), first)); first = False
        if args.limit and len(frames) >= args.limit:
            break
    if args.limit:
        frames = frames[:args.limit]
    # drop any unreadable (e.g. mid-transfer) frames up front so first-flags stay valid
    frames = [(p, fst) for (p, fst) in frames if cv2.imread(p) is not None] or frames
    n = len(frames)
    if n == 0:
        raise SystemExit(f"No frames under {img_root}")
    print(f"[*] stream: {n} frames over {len(seqs)} sequences ({args.split})")

    # ---- live BASE/SUPER anchors on THIS device (includes router cost) ----
    warm = loader(frames[0][0])
    for _ in range(args.warmup):
        backend.run_frame(warm, "base"); backend.run_frame(warm, "super")
    wb = [backend.run_frame(warm, "base")[1] for _ in range(args.warmup)]
    ws = [backend.run_frame(warm, "super")[1] for _ in range(args.warmup)]
    l_base = float(np.median(wb))
    l_super = float(np.median(ws))
    lo = l_base + 0.10 * (l_super - l_base)
    hi = l_super - 0.10 * (l_super - l_base)
    print(f"[*] measured anchors: BASE={l_base:.2f} SUPER={l_super:.2f} ms  "
          f"band [{lo:.2f}, {hi:.2f}] ms")

    # tau bounds: probe the live Ahat range over a prefix so the controller can
    # always reach either extreme (tau>=hi -> all base, tau<=lo -> all super).
    probe = []
    for p, _ in frames[:min(300, n)]:
        x = loader(p)
        probe.append(backend.run_frame(x, "base")[0])
        probe.append(backend.run_frame(x, "super")[0])
    TAU_HI, TAU_LO = float(max(probe)) + 0.03, float(min(probe)) - 0.03
    print(f"[*] tau range: [{TAU_LO:.3f}, {TAU_HI:.3f}]")

    # ---- run the live closed loop for each budget ----
    cells = {}
    print(f"\n{'budget':<10}{'SUPER%':>9}{'mJ/frame':>10}{'W(IN)':>8}{'wall(s)':>9}")
    for bkind in ("step", "sawtooth"):
        Ltgt = target_schedule(bkind, n, lo, hi)
        t0 = time.perf_counter()
        realized, super_rate, e_in, e_cgc, is_super, switched = online_loop(
            backend, frames, loader, Ltgt, l_base, TAU_HI, TAU_LO, args.kp, args.ki, args.beta,
            power=power if power.available else None)
        mj_frame = float(np.mean(e_in))
        # mean power (W) over the run = total energy (mJ) / total latency (ms)
        w_in = float(np.sum(e_in) / max(np.sum(realized), 1e-9))
        # Engine-switch overhead audit: compare frames that *changed* engine vs frames
        # that stayed on the same engine, separately per path (both engines are
        # co-resident, so a switch is just selecting the other context -- no load).
        sw = switched & is_super; st = (~switched) & is_super
        sw_b = switched & (~is_super); st_b = (~switched) & (~is_super)
        def _m(mask): return float(realized[mask].mean()) if mask.any() else float("nan")
        switch_audit = {
            "super_switch_ms": _m(sw), "super_steady_ms": _m(st),
            "super_n_switch": int(sw.sum()), "super_n_steady": int(st.sum()),
            "base_switch_ms": _m(sw_b), "base_steady_ms": _m(st_b),
            "base_n_switch": int(sw_b.sum()), "base_n_steady": int(st_b.sum()),
            "n_switch": int(switched.sum())}
        cells[bkind] = {"super_rate": super_rate,
                        "switch_audit": switch_audit,
                        "realized": realized.tolist(),
                        "target": Ltgt.tolist(),
                        "energy_in_mj": e_in.tolist(),
                        "energy_cgc_mj": e_cgc.tolist(),
                        "mean_mj_in": mj_frame,
                        "mean_mj_cgc": float(np.mean(e_cgc)),
                        "total_j_in": float(np.sum(e_in) / 1000.0),
                        "mean_W_in": w_in,
                        "mean_W_cgc": float(np.sum(e_cgc) / max(np.sum(realized), 1e-9))}
        wall = time.perf_counter() - t0
        print(f"{bkind:<10}{super_rate * 100:>8.1f}%{mj_frame:>10.1f}{w_in:>8.2f}{wall:>9.1f}",
              flush=True)
        sa = switch_audit
        print(f"   switch audit ({sa['n_switch']} switches): "
              f"SUPER switch={sa['super_switch_ms']:.2f} vs steady={sa['super_steady_ms']:.2f} ms | "
              f"BASE switch={sa['base_switch_ms']:.2f} vs steady={sa['base_steady_ms']:.2f} ms",
              flush=True)

    dump = {"l_base": l_base, "l_super": l_super, "win": args.win,
            "kp": args.kp, "ki": args.ki, "beta": args.beta,
            "device_name": dev_name, "imgsz": args.imgsz, "n_frames": n,
            "policy": args.policy, "feat": backend.feat,
            "power_rails": list(power.chans), "cells": cells}
    Path(args.dump_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dump, open(args.dump_out, "w"))
    print(f"[*] dump -> {args.dump_out}")
    render(dump, args.win, args.out)


if __name__ == "__main__":
    main()
