"""Tracklet state machine (paper Sec. 2.4 / A.6): tentative -> confirmed -> deleted.

Confirm threshold N = tau + 1 (default 4, from A.8's tau=3): a tentative
tracklet is promoted to confirmed only after N consecutive detections.
Deletion: an unmatched *unconfirmed* tracklet is deleted immediately; an
unmatched *confirmed* tracklet is deleted only after 100 consecutive misses
(A.8's default). These numbers are exposed as constructor args rather than
hardcoded, since A.8 ties N to tau and different tau values are explored in
the paper's own ablation (Table 3).
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from .kalman_filter import KalmanFilter, xyah_to_xywh, xywh_to_xyah

DEFAULT_CONFIRM_HITS = 4  # N = tau + 1, tau = 3 (paper default)
DEFAULT_MAX_AGE = 100  # paper A.8


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


class Track:
    """One tracklet: Kalman state, lifecycle state, and embedding history.

    ``embedding_history`` accumulates one embedding per successful update,
    in frame order -- the per-target sequence Phase 8's cascade matching
    (and, historically, Phase 6's MSFP) consumes.
    """

    def __init__(
        self,
        track_id: int,
        mean: np.ndarray,
        covariance: np.ndarray,
        confirm_hits: int = DEFAULT_CONFIRM_HITS,
        max_age: int = DEFAULT_MAX_AGE,
        embedding: np.ndarray | None = None,
    ) -> None:
        self.track_id = track_id
        self.mean = mean
        self.covariance = covariance
        self.confirm_hits = confirm_hits
        self.max_age = max_age

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE

        self.embedding_history: list[np.ndarray] = []
        if embedding is not None:
            self.embedding_history.append(embedding)

    @classmethod
    def initiate(
        cls,
        kalman_filter: KalmanFilter,
        track_id: int,
        bbox_xywh: np.ndarray,
        confirm_hits: int = DEFAULT_CONFIRM_HITS,
        max_age: int = DEFAULT_MAX_AGE,
        embedding: np.ndarray | None = None,
    ) -> "Track":
        """Start a new tentative track from a single detection."""

        mean, covariance = kalman_filter.initiate(xywh_to_xyah(np.asarray(bbox_xywh, dtype=np.float64)))
        return cls(track_id, mean, covariance, confirm_hits=confirm_hits, max_age=max_age, embedding=embedding)

    @property
    def bbox_xywh(self) -> np.ndarray:
        """Current [left, top, width, height] box estimate."""

        return xyah_to_xywh(self.mean[:4])

    def predict(self, kalman_filter: KalmanFilter) -> None:
        """Advance one frame without a matching detection.

        Called once per frame for every live track (matched or not) before
        association; ``update``/``mark_missed`` then resolve the outcome.
        """

        self.mean, self.covariance = kalman_filter.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kalman_filter: KalmanFilter, bbox_xywh: np.ndarray, embedding: np.ndarray | None = None) -> None:
        """Correct the track with a matched detection, and progress its state."""

        measurement = xywh_to_xyah(np.asarray(bbox_xywh, dtype=np.float64))
        self.mean, self.covariance = kalman_filter.update(self.mean, self.covariance, measurement)
        self.hits += 1
        self.time_since_update = 0

        if embedding is not None:
            self.embedding_history.append(embedding)

        if self.state is TrackState.TENTATIVE and self.hits >= self.confirm_hits:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Apply the paper's deletion rules for an unmatched track this frame."""

        if self.state is TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    def is_tentative(self) -> bool:
        return self.state is TrackState.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state is TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state is TrackState.DELETED
