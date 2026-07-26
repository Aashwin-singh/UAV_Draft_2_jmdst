"""Tracking infrastructure for the JMDST reproduction (Phases 7-8)."""

from .kalman_filter import KalmanFilter, xyah_to_xywh, xywh_to_xyah
from .matching import iou_xywh
from .track import Track, TrackState
from .tracker import (
    Detection,
    Tracker,
    TrackingOutput,
    boundary_distance,
    expansion_targets,
    filter_tracking_outputs,
)

__all__ = [
    "KalmanFilter",
    "Track",
    "TrackState",
    "xyah_to_xywh",
    "xywh_to_xyah",
    "iou_xywh",
    "Detection",
    "Tracker",
    "TrackingOutput",
    "boundary_distance",
    "expansion_targets",
    "filter_tracking_outputs",
]
