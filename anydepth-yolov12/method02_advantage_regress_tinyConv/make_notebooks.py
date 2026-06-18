"""Generate organized Jupyter notebooks that orchestrate the (otherwise sprawling)
scripts in this directory. The scripts remain the source of truth; each notebook is a
documented, runnable entry point that calls them with the correct arguments.

    python method02_advantage_regress_tinyConv/make_notebooks.py

Run notebooks with a kernel in the `yolov12` conda env, from anywhere (the setup cell
chdir's to the repo root). Commands use `{PY}` = the kernel's own interpreter.
"""
import json
from pathlib import Path

PKG = "method02_advantage_regress_tinyConv"
OUT = f"{PKG}/outputs"
NB_DIR = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


SETUP = code(
    "# --- setup: run from the repo root with the kernel's own python ---\n"
    "import os, sys\n"
    "while not os.path.isdir(f'{os.getcwd()}/" + PKG + "'):\n"
    "    os.chdir('..')          # walk up to the anydepth-yolov12 repo root\n"
    "PY = sys.executable\n"
    "print('cwd =', os.getcwd()); print('python =', PY)")


def notebook(intro_md, cells):
    nb = {"cells": [md(intro_md), SETUP] + cells,
          "metadata": {"kernelspec": {"display_name": "yolov12", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return nb


# ----------------------------------------------------------------------------------
NOTEBOOKS = {}

NOTEBOOKS["0_data_and_training"] = notebook(
    "# 0 · Data caches & router training\n\n"
    "Builds the per-image feature/loss caches (the frozen detector forwarded along the\n"
    "BASE and SUPER paths) and trains the TinyConv advantage-regression router on them.\n"
    "Everything downstream consumes these caches / router checkpoints.\n\n"
    "- `build_cache.py` — dumps `input/pred` feature grids + `loss_base/loss_super` per image.\n"
    "- `train_policy.py` — regresses the advantage A = L_base − L_super; selects the\n"
    "  checkpoint with min validation MSE (`--select val_mse`). 5 seeds.",
    [md("## Build caches (KITTI / BDD100K)\nSkip if `outputs/<ds>/cache_{train,val}_g2.pt` already exist."),
     code(f"# KITTI cache (grid 2x2). Heavy: forwards the detector over all images, both paths.\n"
          f"!{{PY}} -m {PKG}.build_cache --help"),
     md("## Train the router (5 seeds, min-val-MSE checkpoint)"),
     code(f"# Example (BDD100K, feat=input). See run_*.sh for the exact sweeps used.\n"
          f"!{{PY}} -m {PKG}.train_policy --dataset bdd100k --feat input --norm batch \\\n"
          f"    --cache {OUT}/bdd100k/cache_train_g2.pt --val_cache {OUT}/bdd100k/cache_val_g2.pt \\\n"
          f"    --epochs 30 --batch 256 --lr 1e-3 --select val_mse \\\n"
          f"    --out {OUT}/bdd100k/policy_scenario_s0.pt")])

NOTEBOOKS["main_result"] = notebook(
    "# Main result · Pareto dominance (Fig. 5) & device efficiency (Table 3)\n\n"
    "The single trained router traces a continuous accuracy–efficiency Pareto front that\n"
    "dominates the heuristic baselines on KITTI, BDD100K and Waymo, and Table 3 reports the\n"
    "RTX 3090 latency / FPS / energy across operating points.\n\n"
    "AP axes are plotted in **percent** (e.g. 22.4, not 0.224).",
    [md("## Fig. 5 — Pareto curves (AP@[.50:.95] vs SUPER usage)\n"
        "`make_main_figure.py`: learned router (mean±std, 5 seeds) vs random / luminance /\n"
        "edge-density / confidence baselines, with a GFLOPs top axis."),
     code(f"# KITTI (both-feature router)\n"
          f"!{{PY}} -m {PKG}.make_main_figure \\\n"
          f"    --curve {OUT}/kitti/eval/video_curve_main_both_g2.json \\\n"
          f"    --policy_fam 'policy_both_s(\\d+)' --out paper/fig_main_kitti_ap5095.pdf"),
     code(f"# BDD100K\n"
          f"!{{PY}} -m {PKG}.make_main_figure \\\n"
          f"    --curve {OUT}/bdd100k/eval/video_curve_full_aligned.json \\\n"
          f"    --policy_fam 'policy_bn(\\d+)' --out paper/fig_main_bdd_ap5095.pdf"),
     code(f"# Waymo Open\n"
          f"!{{PY}} -m {PKG}.make_main_figure \\\n"
          f"    --curve {OUT}/waymo/eval_both/video_curve.json \\\n"
          f"    --policy_fam 'policy_seed(\\d+)' --out paper/fig_main_waymo_ap5095.pdf"),
     md("## Table 3 — on-device efficiency (RTX 3090, n=1000)\n"
        "`bench_device.py` measures BASE/SUPER latency·energy anchors + router overhead\n"
        "identically for all three datasets, then blends by each operating point's SUPER%."),
     code(f"!{{PY}} -m {PKG}.bench_device --n 1000 --warmup 40"),
     md("## Underlying video routing curves\n"
        "`eval_video_{bdd,waymo}.py` produce the `video_curve*.json` consumed above\n"
        "(streaming AP-vs-FLOPs over the MOT/tracking videos). Heavy — usually run via the\n"
        "`run_*.sh` / sbatch scripts; shown here for reference."),
     code(f"!{{PY}} -m {PKG}.eval_video_bdd --help")])

NOTEBOOKS["ablation_study"] = notebook(
    "# Ablation study\n\n"
    "Design choices for the router: **feature source** (backbone / neck / both),\n"
    "**router architecture & grid** (GAP-MLP vs TinyConv 2×2 / 8×8), and the\n"
    "**previous-action sampling ratio** p(super) used during training.",
    [md("## Feature-source ablation (Fig. 9) — BDD100K\n`make_feat_ablation_figure.py`"),
     code(f"!{{PY}} -m {PKG}.make_feat_ablation_figure \\\n"
          f"    --curve {OUT}/bdd100k/eval/video_curve_full3_aligned.json \\\n"
          f"    --out paper/fig_abl_feat_ap5095.pdf"),
     md("## Router design & grid resolution (Fig. 10) — KITTI\n"
        "GAP-MLP vs TinyConv 2×2 / 8×8. See also `plot_arch_compare.py`,\n"
        "`plot_arch_overlay.py`, `plot_norm_compare.py` for the BatchNorm/LayerNorm and\n"
        "per-input arch comparisons."),
     code(f"!{{PY}} -m {PKG}.plot_arch_overlay --help"),
     md("## Previous-action sampling ratio p(super) (Fig. 11)\n"
        "Balanced (p=0.5) vs all-base / all-super context during training.\n"
        "`plot_ablation.py` aggregates the seed sweep."),
     code(f"!{{PY}} -m {PKG}.plot_ablation --help")])

NOTEBOOKS["router_analysis"] = notebook(
    "# Router behavior analysis\n\n"
    "What does the router actually predict, and is the advantage signal learnable?\n\n"
    "- **Calibration (Fig. 6)** — predicted Â vs true A across deciles (KITTI/BDD).\n"
    "- **Decile behavior** — interpretable scene attributes of high/low-Â frames.\n"
    "- **Negative-advantage** — does the router flag frames where SUPER *hurts* (A<0)?\n"
    "- **Advantage distribution / loss-gap** — motivation for regressing A.",
    [md("## Fig. 6 — calibration / reliability of Â vs A\n"
        "`make_router_analysis.py` (KITTI + BDD panels; add Waymo once its val cache +\n"
        "policy exist: `cache_val_both.pt`, `policy_both_s0.pt`)."),
     code(f"!{{PY}} -m {PKG}.make_router_analysis --out_dir paper"),
     md("## Decile scene-attribute behavior (BDD100K)\n`make_router_behavior_bdd.py`"),
     code(f"!{{PY}} -m {PKG}.make_router_behavior_bdd \\\n"
          f"    --cache {OUT}/bdd100k/cache_val_g2.pt \\\n"
          f"    --policy {OUT}/bdd100k/policy_scenario_s0.pt"),
     md("## Negative-advantage analysis (does it catch A<0?)\n`analyze_negative_advantage.py`"),
     code(f"!{{PY}} -m {PKG}.analyze_negative_advantage --dataset bdd100k \\\n"
          f"    --cache {OUT}/bdd100k/cache_val_g2.pt \\\n"
          f"    --policy {OUT}/bdd100k/policy_scenario_s0.pt"),
     code(f"!{{PY}} -m {PKG}.analyze_negative_advantage --dataset kitti \\\n"
          f"    --cache {OUT}/kitti/cache_val_g2.pt --policy {OUT}/kitti/policy.pt"),
     md("## Advantage distribution & loss-gap (motivation)\n"
        "`plot_advantage_apgap.py`, `analyze_lossgap.py`, `analyze_super_usage.py`."),
     code(f"!{{PY}} -m {PKG}.plot_advantage_apgap --help")])

NOTEBOOKS["budget_control"] = notebook(
    "# Runtime latency-budget control (PI)\n\n"
    "A PI controller adjusts the routing threshold τ frame-by-frame to track a\n"
    "time-varying latency budget L*(t), as the scene condition drifts.\n\n"
    "- **Live on-device demo (Fig. 8)** — closed loop on the RTX 3090, measured per frame.\n"
    "- **Scenario figures** — step / sawtooth budgets across condition-drift sequences.\n"
    "- **Gain sensitivity** — robustness of tracking to the PI gains.",
    [md("## Fig. 8 — live on-device budget tracking (RTX 3090)\n"
        "`online_budget_demo.py` runs the full closed loop live (one depth per frame,\n"
        "latency measured on the fly). `--replot --win N` re-smooths from the saved dump."),
     code(f"!{{PY}} -m {PKG}.online_budget_demo \\\n"
          f"    --weight finetuned_bdd100k/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-720_singleScale_augNothing_alpha0.2_orig_mAP35.1_33.8.pt \\\n"
          f"    --policy {OUT}/bdd100k/policy_scenario_s0.pt"),
     code(f"# re-render only (no inference): change the display smoothing window\n"
          f"!{{PY}} -m {PKG}.online_budget_demo --replot --win 30 --weight x --policy x \\\n"
          f"    --dump {OUT}/bdd100k/online_budget_demo.json \\\n"
          f"    --out {OUT}/bdd100k/online_budget_demo.pdf"),
     md("## Condition-drift scenario figures (replay simulator)\n`make_scenario_figures_bdd.py`, `sim_latency_budget.py`"),
     code(f"!{{PY}} -m {PKG}.make_scenario_figures_bdd --help"),
     md("## PI gain sensitivity & tracking plots\n`sweep_pi_gains.py`, `plot_pi_tracking.py`"),
     code(f"!{{PY}} -m {PKG}.sweep_pi_gains --help")])

NOTEBOOKS["motivation_and_conditions"] = notebook(
    "# Motivation & condition-stratified analysis\n\n"
    "- **Teaser (Fig. 1)** — visually-hard scenes where the deep path adds nothing.\n"
    "- **Condition-stratified AP (Table 1)** — SUPER−BASE gap by time-of-day / weather /\n"
    "  scene / crowd, showing the benefit is input-dependent but not explained by coarse\n"
    "  semantic categories.",
    [md("## Fig. 1 — motivation sample figure (BDD100K)\n`make_sample_figure_bdd.py`"),
     code(f"!{{PY}} -m {PKG}.make_sample_figure_bdd \\\n"
          f"    --picks 'b4065fc4-ede06556,b216243d-55963da2' --out paper/fig_sample_routing.pdf"),
     md("## Table 1 — condition-stratified BASE vs SUPER AP\n"
        "`eval_condition_stratified_bdd.py` → json → `make_condition_table.py` → LaTeX."),
     code(f"!{{PY}} -m {PKG}.eval_condition_stratified_bdd --help"),
     code(f"!{{PY}} -m {PKG}.make_condition_table --help")])

NOTEBOOKS["device_and_trt"] = notebook(
    "# Device benchmarking & TensorRT deployment\n\n"
    "Eager (FP32) anchors for Table 3, and the TensorRT FP16 two-engine deployment of\n"
    "depth routing: BASE and SUPER are compiled as separate static engines (TRT cannot\n"
    "skip layers within one engine), and the router selects which to run per frame. The\n"
    "engines also emit the intermediate `tap` feature maps the router consumes.",
    [md("## Eager device anchors (Table 3)\n`bench_device.py` (KITTI/BDD/Waymo, n=1000)."),
     code(f"!{{PY}} -m {PKG}.bench_device --n 1000 --warmup 40"),
     md("## TensorRT: export → build → benchmark\n"
        "Scripts live in `trt_bench/`. Order: PyTorch → ONNX → TRT engine (FP16).\n"
        "Each detector path is exported with detection **and** raw tap maps (layers 4,6,8)\n"
        "so the router can run on-device."),
     code("# 1) export BASE/SUPER paths to ONNX (skip baked in) + raw tap maps\n"
          "!{PY} trt_bench/export_onnx.py --weight finetuned_bdd100k/anydepth_best.pt \\\n"
          "    --imgsz 720 1280 --grid 2 --out_dir trt_bench/onnx/bdd"),
     code("# 2) build FP16 engines\n"
          "!{PY} trt_bench/build_engine.py --onnx trt_bench/onnx/bdd/base.onnx --fp16\n"
          "!{PY} trt_bench/build_engine.py --onnx trt_bench/onnx/bdd/super.onnx --fp16"),
     code("# 3a) pure-engine latency + switching overhead\n"
          "!{PY} trt_bench/bench_trt.py --base trt_bench/onnx/bdd/base.fp16.engine \\\n"
          "    --super trt_bench/onnx/bdd/super.fp16.engine --iters 1000 --warmup 100"),
     code("# 3b) real-video threshold sweep: end-to-end realized SUPER% / latency / FPS / energy\n"
          "!{PY} trt_bench/trt_video_eval.py --base trt_bench/onnx/bdd/base.fp16.engine \\\n"
          "    --super trt_bench/onnx/bdd/super.fp16.engine \\\n"
          f"    --policy {OUT}/bdd100k/policy_scenario_s0.pt \\\n"
          "    --mot_root /media/data/bdd100k_mot/val --imgsz 720 1280 --limit 20")])


def main():
    for name, nb in NOTEBOOKS.items():
        # {PY} inside trt cells is literal in those f-strings? ensure it stays a placeholder
        path = NB_DIR / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print(f"[*] wrote {path.relative_to(NB_DIR.parent)}")


if __name__ == "__main__":
    main()
