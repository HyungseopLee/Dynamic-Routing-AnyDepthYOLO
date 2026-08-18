"""Shim that loads eval_baseline_kitti from its compiled cache.

The original eval_baseline_kitti.py was removed during cleanup but the .pyc
remains. All eval scripts import this module as B instead of eval_baseline_kitti.
"""
import importlib.util
import sys
from pathlib import Path

_pyc = Path(__file__).resolve().parents[2] / "__pycache__/eval_baseline_kitti.cpython-311.pyc"
if not _pyc.exists():
    raise ImportError(
        f"eval_baseline_kitti.pyc not found at {_pyc}. "
        "The compiled cache is required for AP matching and mAP computation."
    )

_spec = importlib.util.spec_from_file_location("eval_baseline_kitti", str(_pyc))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["eval_baseline_kitti"] = _mod
_spec.loader.exec_module(_mod)

# re-export everything as module-level attributes so callers can do:
#   import method_advantage_regress.eval.eval_utils as B
#   B.EVAL_CLS = ...   (mutation works because we update this module's __dict__)
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v


def __getattr__(name):
    return getattr(_mod, name)


def __setattr__(name, value):  # propagate mutations (e.g. B.EVAL_CLS = ...) back to _mod
    setattr(_mod, name, value)
    globals()[name] = value
