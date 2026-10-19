"""Tracking infrastructure (Phases 7-8)."""

from .cmc import CameraMotionCompensator
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
    "CameraMotionCompensator",
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
