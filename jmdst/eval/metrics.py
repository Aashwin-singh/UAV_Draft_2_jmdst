"""CLEAR MOT + IDF1 metrics via motmetrics (paper Sec. 3.2 Eq. 10-12).

Wraps py-motmetrics to compute MOTA, MOTP, IDF1, ID switches, FP and FN per
sequence and overall. MOTP is converted to the paper's convention (mean IoU
of matches) since motmetrics reports the mean IoU-*distance* (1 - IoU).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# py-motmetrics 1.4.0 calls np.asfarray, removed in NumPy 2.0. Restore it
# before importing motmetrics so distance/metric computations work.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import logging

import motmetrics as mm

# motmetrics logs INFO timing lines per compute() call; quiet it for clean output.
logging.getLogger("motmetrics").setLevel(logging.WARNING)

from .io import FrameDict


@dataclass
class ClearMotResult:
    mota: float
    motp: float  # paper convention: mean IoU of matches (higher is better)
    idf1: float
    id_switches: int
    false_positives: int
    false_negatives: int
    num_objects: int


def _iou_distance(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    # motmetrics iou_matrix: 1 - IoU, with entries above the distance cutoff
    # (i.e. IoU below iou_threshold) set to NaN so they cannot be matched.
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))
    return mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0 - iou_threshold)


def _accumulate(gt: FrameDict, pred: FrameDict, iou_threshold: float) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=False)
    for frame_id in sorted(set(gt) | set(pred)):
        gt_objs = gt.get(frame_id, [])
        pred_objs = pred.get(frame_id, [])
        gt_ids = [i for i, _ in gt_objs]
        pred_ids = [i for i, _ in pred_objs]
        gt_boxes = np.array([b for _, b in gt_objs], dtype=np.float64).reshape(-1, 4)
        pred_boxes = np.array([b for _, b in pred_objs], dtype=np.float64).reshape(-1, 4)
        acc.update(gt_ids, pred_ids, _iou_distance(gt_boxes, pred_boxes, iou_threshold), frameid=frame_id)
    return acc


def evaluate_clearmot(
    ground_truth: dict[str, FrameDict],
    predictions: dict[str, FrameDict],
    iou_threshold: float = 0.5,
) -> dict[str, ClearMotResult]:
    """Compute per-sequence and OVERALL CLEAR MOT + IDF1 metrics.

    Returns a dict keyed by sequence name plus an ``"OVERALL"`` entry.
    """

    accs, names = [], []
    for name in sorted(ground_truth):
        accs.append(_accumulate(ground_truth[name], predictions.get(name, {}), iou_threshold))
        names.append(name)

    metrics = ["mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses", "num_objects"]
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

    results: dict[str, ClearMotResult] = {}
    for name, row in summary.iterrows():
        motp_distance = row["motp"]
        results[name] = ClearMotResult(
            mota=float(row["mota"]),
            motp=float(1.0 - motp_distance) if np.isfinite(motp_distance) else float("nan"),
            idf1=float(row["idf1"]),
            id_switches=int(row["num_switches"]),
            false_positives=int(row["num_false_positives"]),
            false_negatives=int(row["num_misses"]),
            num_objects=int(row["num_objects"]),
        )
    return results
