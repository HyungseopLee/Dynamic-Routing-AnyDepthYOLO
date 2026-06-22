Fig 8 (latency-budget tracking) reproduction package for Jetson
================================================================

Contents
--------
bdd100k_mot/val/videos/   11 scenario .mov clips used by Fig 8 (~220 MB)
bdd100k_mot/val/labels/   matching MOT label .json for each clip
assets/scenarios.json     scenario family definitions (night/day, city/highway, clear/rainy)
assets/policy_scenario_s0.pt   the single routing policy (feat=input) used for the figure
assets/anydepth_bdd_alpha0.2.pt  the frozen AnyDepth-YOLOv12s detector weight

This is everything Fig 8 needs -- NOT the full BDD100K dataset.

How to run on the Jetson
------------------------
Place this package so the repo can find the data, e.g.:

  /media/data/bdd100k_mot/val/{videos,labels}     <- copy from bdd100k_mot/val here
  (or pass --mot_root <path-to>/bdd100k_mot/val)

Then run the online budget demo (end-to-end: live BASE/SUPER latency anchors are
measured on-device, the PI loop runs, and the figure is rendered):

  python -m method02_advantage_regress_tinyConv.online_budget_demo \
    --weight <path>/anydepth_bdd_alpha0.2.pt \
    --policy <path>/policy_scenario_s0.pt \
    --scenarios <path>/scenarios.json \
    --mot_root <path>/bdd100k_mot/val \
    --dump online_budget_demo_jetson.json \
    --out fig8_jetson.pdf

Notes
-----
- BASE/SUPER per-frame latency is measured live on the Jetson during warm-up, so the
  budget band and PI tracking reflect Jetson hardware automatically.
- Display smoothing is a centered (zero-phase) moving average, window = 30 frames
  (--win 30, default). To re-render without re-running inference:
    python -m method02_advantage_regress_tinyConv.online_budget_demo --replot \
      --dump online_budget_demo_jetson.json --out fig8_jetson.pdf --win 30
- The policy is feat=input for BDD (not feat=both).
