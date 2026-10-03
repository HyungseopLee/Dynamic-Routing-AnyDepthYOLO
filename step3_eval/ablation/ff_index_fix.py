"""Offline test: does the feedforward term's time index matter?

Comment-thread finding: in step4_deploy/online_budget_demo_stream.py, the threshold
tau(t+1) -- which routes frame t+1 -- is built from tau_ff[t] (interpolated from
C_target(t), the CURRENT frame's own target) combined with e(t) (feedback from the
current frame's own error). Since C_target is a known schedule (not something that
needs predicting), tau_ff could instead use C_target(t+1) -- the target the
threshold will actually be applied against -- which is already known in advance.
This script replays the exact EMA + PI + anti-windup + feedforward control loop from
online_budget_demo_stream.py entirely offline, driven by REAL per-frame advantage
values (ah_s0_base/super) from results/step3_eval/bdd100k/eval/frames.npz, under a
synthetic step/sawtooth target schedule. Both index conventions are run on the exact
same Ahat sequence and target profile so the only difference is which C_target the
feedforward term maps.

Usage (run from repo root):
    python -m step3_eval.ablation.ff_index_fix
"""
import numpy as np

DUMP = "results/step3_eval/bdd100k/eval/frames.npz"
# BDD100K Jetson-scale latency anchors (ms), matching Table 3 (b) BDD100K rows.
L_BASE = 24.99
L_SUPER = 41.64
KP, KI, BETA = 2.0, 0.33, 0.85
WARMUP = 60
N_SEED = 5


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


def run_loop(ahat, l_base, l_super, ctrl, shift_ff: bool, kp=KP, ki=KI, beta=BETA,
             warmup=WARMUP, seed=0):
    """Faithful replay of online_budget_demo_stream's control loop.

    `ahat[i]` is the router's advantage estimate "as if frame i's features came from
    whichever path was NOT taken" is unavailable offline, so -- matching the online
    system's own causal design (Eq. route: config(t) uses Ahat(F_{t-1})) -- we use a
    single per-frame Ahat value (the seed's base-path advantage trace) as the signal
    the router would have produced causally; the realized cost is l_super if routed
    super else l_base (hardware latency is ~constant per path, confirmed by the
    tight clustering of real Jetson 'realized' logs around the two anchors).

    shift_ff=False reproduces the current code (tau_ff[t] paired with e(t)).
    shift_ff=True uses tau_ff[t+1] (the next frame's OWN target) instead.
    """
    n = len(ahat)
    rng = np.random.default_rng(seed)
    wa = ahat[:warmup]
    TAU_LO, TAU_HI = float(wa.min()) - 0.03, float(wa.max()) + 0.03
    tau0 = 0.5 * (TAU_LO + TAU_HI)

    def pct_to_tau(pct):
        return float(np.clip(np.quantile(wa, 1.0 - pct), TAU_LO, TAU_HI))

    ff_pct = np.clip((ctrl - l_base) / max(l_super - l_base, 1e-6), 0.0, 1.0)
    tau_ff = np.array([pct_to_tau(p) for p in ff_pct])

    ctrl_span = max(l_super - l_base, 1e-6)
    gscale = (TAU_HI - TAU_LO) / ctrl_span
    kp_, ki_ = kp * gscale, ki * gscale

    config = "base"; tau = tau0; integ = 0.0
    sig_ema = float(ctrl[0])
    realized = np.empty(n)
    for t in range(n):
        lat = l_super if config == "super" else l_base
        realized[t] = lat
        sig_ema = beta * sig_ema + (1 - beta) * lat
        if t > 0:
            delta = ctrl[t] - ctrl[t - 1]
            if delta < -1.0:
                integ = 0.0
        e = ctrl[t] - sig_ema
        integ += e
        ff_idx = min(t + 1, n - 1) if shift_ff else t
        tau_un = tau_ff[ff_idx] - kp_ * e - ki_ * integ
        tau = float(np.clip(tau_un, TAU_LO, TAU_HI))
        integ += (tau_un - tau) / ki_
        config = "super" if ahat[t] > tau else "base"
    return realized


def main():
    z = np.load(DUMP, allow_pickle=False)
    seq = z["seq"]
    starts = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1]])
    ends = np.r_[starts[1:], len(seq)]
    lens = ends - starts
    s, e = starts[np.argmax(lens)], ends[np.argmax(lens)]
    print(f"[*] using longest sequence '{seq[s]}', n={e - s} frames")

    for kind in ("step", "sawtooth"):
        for shift_ff, label in ((False, "tau_ff(t)  [current code]"),
                                (True, "tau_ff(t+1) [fixed]      ")):
            maes = []
            for seed in range(N_SEED):
                ahat = z[f"ah_s{seed}_base"][s:e]
                n = len(ahat)
                lo = L_BASE + 0.10 * (L_SUPER - L_BASE)
                hi = L_SUPER - 0.10 * (L_SUPER - L_BASE)
                ctrl = target_schedule(kind, n, lo, hi)
                realized = run_loop(ahat, L_BASE, L_SUPER, ctrl, shift_ff=shift_ff, seed=seed)
                maes.append(float(np.mean(np.abs(realized - ctrl))))
            print(f"  {kind:<9} {label}: MAE = {np.mean(maes):.3f} ms (std {np.std(maes):.3f}, 5 seeds)")


if __name__ == "__main__":
    main()
