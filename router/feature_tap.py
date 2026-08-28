"""Feature tap for the 2-level depth router.

Stateless (no learnable params): decides *which* detector layers feed the
router and slices the detector's `return_features=True` output into the two
context groups. All learnable projection/fusion lives in RouterNetwork.

  - input-level context : GAP of backbone layers {4, 6, 8}  (S2/S3/S4 == P3/P4/P5)
  - pred-level  context : GAP of neck-fusion layers {14, 17, 20} (Detect inputs)

The paper uses BOTH groups (`--feat both`); the input-only and pred-only subsets
exist for the feature ablation in step3_eval/ablation/.

The pooled (GAP) vectors are produced by the detector's
`_predict_once(..., return_features=True)`, which returns one [B, C] vector per
layer in `model.save`. Make sure {4,6,8,14,17,20} are all in `model.save`.
"""

from typing import List, Sequence

import torch

INPUT_LEVEL_LAYERS = (4, 6, 8)      # backbone taps
PRED_LEVEL_LAYERS = (14, 17, 20)    # neck taps
STATE_LAYERS = INPUT_LEVEL_LAYERS + PRED_LEVEL_LAYERS  # (4,6,8,14,17,20)

# --feat is spelled after the tap location (backbone/neck); "input"/"pred" are the
# original internal names, kept as accepted aliases so older checkpoints -- which
# store the raw string in ckpt["args"] -- still load. normalize_feat() maps the
# user-facing spelling onto the internal one, which stays "input"/"pred"/"both".
FEAT_CHOICES = ("backbone", "neck", "both", "input", "pred")
_FEAT_CANON = {"backbone": "input", "neck": "pred", "both": "both",
               "input": "input", "pred": "pred"}


def normalize_feat(feat: str) -> str:
    """Map a user-facing --feat value onto the internal one."""
    try:
        return _FEAT_CANON[feat]
    except KeyError:
        raise ValueError(f"unknown --feat {feat!r}; choose from {FEAT_CHOICES}")


def split_features(features: List[torch.Tensor], save: Sequence[int]):
    """Slice the `return_features=True` output into (input_feats, pred_feats).

    `_predict_once` appends one pooled vector per layer whose index is in
    `model.save`, in ascending layer-index (loop) order -- so feature[k]
    corresponds to the k-th smallest index in `save`.
    """
    order = sorted(set(save))
    idx = {layer: features[order.index(layer)] for layer in STATE_LAYERS}
    input_feats = [idx[l] for l in INPUT_LEVEL_LAYERS]
    pred_feats = [idx[l] for l in PRED_LEVEL_LAYERS]
    return input_feats, pred_feats
