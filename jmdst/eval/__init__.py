"""Tracking evaluation (Phase 10): CLEAR MOT + IDF1 + HOTA."""

from .hota import HotaResult, compute_hota
from .io import (
    filter_predictions_by_ignore,
    load_ground_truth,
    load_ignore_regions,
    load_mot_predictions,
)
from .metrics import ClearMotResult, evaluate_clearmot

__all__ = [
    "load_ground_truth",
    "load_mot_predictions",
    "load_ignore_regions",
    "filter_predictions_by_ignore",
    "evaluate_clearmot",
    "ClearMotResult",
    "compute_hota",
    "HotaResult",
]
