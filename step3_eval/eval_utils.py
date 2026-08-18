"""Shim re-exporting eval_baseline_kitti (AP matching + mAP computation).

All eval scripts import this module as B instead of eval_baseline_kitti.
"""
import sys

from . import eval_baseline_kitti as _mod

sys.modules["eval_baseline_kitti"] = _mod

# re-export everything as module-level attributes so callers can do:
#   import step3_eval.eval_utils as B
#   B.EVAL_CLS = ...   (mutation works because we update this module's __dict__)
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v


def __getattr__(name):
    return getattr(_mod, name)


def __setattr__(name, value):  # propagate mutations (e.g. B.EVAL_CLS = ...) back to _mod
    setattr(_mod, name, value)
    globals()[name] = value
