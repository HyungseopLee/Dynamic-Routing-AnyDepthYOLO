"""COCO-format dump + pycocotools summarize() for selected routing strategies.

Why this exists: eval_video.py sweeps ~58 strategies in one pass over the video,
so it accumulates matches with its own greedy matcher rather than materialising a
COCO JSON per strategy. That is fast, but it is our own code -- and a reviewer
cannot check it against a reference. This module closes that gap: for a handful
of named strategies it collects detections in COCO format and hands them to
pycocotools, giving the exact 12-line COCOeval.summarize() table and, at the same
time, an independent check of our in-loop AP/AR.

GT conventions:
  - one image per scored frame (ids assigned in evaluation order)
  - KITTI DontCare regions become iscrowd=1 boxes, which is how pycocotools
    ignores detections that land on them (equivalent to filter_dontcare)
  - category ids are our class indices, so per-class numbers line up with
    eval_baseline_kitti.EVAL_CLS
"""
import contextlib
import io
import json
from pathlib import Path

import numpy as np

STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large",
]


class CocoDump:
    """Collects COCO-format GT once and detections for a set of strategies."""

    def __init__(self, strategies, names):
        self.strategies = list(strategies)
        self.images, self.anns = [], []
        self.dets = {s: [] for s in self.strategies}
        self.categories = [{"id": int(c), "name": str(n)} for c, n in sorted(names.items())]
        self._img_id = 0
        self._ann_id = 0

    def add_frame(self, gts, dc, h, w):
        """Register one scored frame; returns its image id."""
        self._img_id += 1
        iid = self._img_id
        self.images.append({"id": iid, "height": int(h), "width": int(w),
                            "file_name": f"{iid:08d}.jpg"})
        for gc, x1, y1, x2, y2 in gts:
            self._add_ann(iid, int(gc), x1, y1, x2, y2, 0)
        for x1, y1, x2, y2 in (dc or []):
            # iscrowd regions are ignored by pycocotools for every category; use the
            # first category id as a placeholder since DontCare has no class.
            self._add_ann(iid, self.categories[0]["id"], x1, y1, x2, y2, 1)
        return iid

    def _add_ann(self, iid, cat, x1, y1, x2, y2, iscrowd):
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        self._ann_id += 1
        self.anns.append({"id": self._ann_id, "image_id": iid, "category_id": cat,
                          "bbox": [float(x1), float(y1), float(w), float(h)],
                          "area": float(w * h), "iscrowd": int(iscrowd)})

    def add_dets(self, strategy, iid, preds):
        """Accumulate one frame's detections as a compact numpy chunk.

        A python dict per detection costs ~500 B; BDD100K has 39.7k frames x 100
        detections x 11 strategies, which would be tens of GB. The 7 float32
        columns here are 28 B per detection, and the dicts pycocotools wants are
        built one strategy at a time in summarize().
        """
        if strategy not in self.dets or not preds:
            return
        a = np.empty((len(preds), 7), dtype=np.float32)
        for i, (c, conf, x1, y1, x2, y2) in enumerate(preds):
            a[i] = (iid, c, x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1), conf)
        self.dets[strategy].append(a)

    def _det_dicts(self, strategy):
        parts = self.dets[strategy]
        if not parts:
            return []
        a = np.concatenate(parts)
        return [{"image_id": int(r[0]), "category_id": int(r[1]),
                 "bbox": [float(r[2]), float(r[3]), float(r[4]), float(r[5])],
                 "score": float(r[6])} for r in a]

    # ── evaluation ────────────────────────────────────────────────────────────
    def gt_dict(self):
        return {"info": {}, "licenses": [], "images": self.images,
                "annotations": self.anns, "categories": self.categories}

    def summarize(self, out_dir=None):
        """Run COCOeval for every collected strategy. Returns {name: {stat: value}}."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "gt.json").write_text(json.dumps(self.gt_dict()))

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO()
            coco_gt.dataset = self.gt_dict()
            coco_gt.createIndex()

        results = {}
        for name in self.strategies:
            dets = self._det_dicts(name)
            if not dets:
                continue
            if out_dir:
                (out_dir / f"dets_{name}.json").write_text(json.dumps(dets))
            with contextlib.redirect_stdout(io.StringIO()):
                coco_dt = coco_gt.loadRes(dets)
                ev = COCOeval(coco_gt, coco_dt, "bbox")
                ev.evaluate(); ev.accumulate()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ev.summarize()
            del dets, coco_dt
            results[name] = {"stats": {k: float(v) for k, v in zip(STAT_NAMES, ev.stats)},
                             "summary": buf.getvalue(),
                             "per_class": _per_class(ev)}
        if out_dir:
            (out_dir / "coco_summary.json").write_text(json.dumps(results, indent=2))
        return results


def _per_class(ev):
    """AP@[.5:.95] and AR@100 per category from an evaluated COCOeval."""
    p = ev.params
    out = {}
    for ki, cat in enumerate(p.catIds):
        prec = ev.eval["precision"][:, :, ki, 0, 2]     # all area, maxDets=100
        rec = ev.eval["recall"][:, ki, 0, 2]
        prec = prec[prec > -1]
        out[str(int(cat))] = {
            "ap": float(np.mean(prec)) if prec.size else float("nan"),
            "ar": float(np.mean(rec[rec > -1])) if (rec > -1).any() else float("nan"),
        }
    return out
