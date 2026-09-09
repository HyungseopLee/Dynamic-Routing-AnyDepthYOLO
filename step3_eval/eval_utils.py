"""Shim exposing eval_baseline_kitti (AP/AR matching + mAP computation).

All eval scripts import this module as B instead of eval_baseline_kitti.

This module *becomes* eval_baseline_kitti: we replace our own entry in
sys.modules with it, so `import step3_eval.eval_utils as B` binds the real
module object. That matters because callers mutate module state --
`B.EVAL_CLS = BDD_EVAL_CLS` -- and the functions that read EVAL_CLS
(match_frame_multi_iou, dataset_map_multi_iou) live in eval_baseline_kitti.

An earlier version re-exported names into this module's globals and tried to
forward writes with a module-level __setattr__. Python does not honour
__setattr__ on modules (PEP 562 covers only __getattr__/__dir__), so those
writes silently stayed here and the eval ran with the KITTI class list on
every dataset -- BDD100K scored 7 classes instead of 8, deflating mAP by 7/8.
"""
import sys

from . import eval_baseline_kitti as _mod

# bare name, for pickles/legacy imports that reference it directly
sys.modules["eval_baseline_kitti"] = _mod
# make `step3_eval.eval_utils` resolve to the real module object
sys.modules[__name__] = _mod
