"""Modified-DeepSORT tracker: detection-frame association + tracking-branch rules.

Implements the paper's update module (Sec. 2.4 / A.6):

* ``Tracker.update`` runs the detection-frame association exactly as A.6 steps
  1-7: cascade (appearance) matching on confirmed tracks, IoU assignment on
  unconfirmed + cascade-unmatched-confirmed tracks (time_since_update <= tau+1),
  Kalman updates, deletions, and new-track initiation.
* ``expansion_targets`` implements the missed-detection expanded tracking set.
* ``filter_tracking_outputs`` implements the tracking-branch output filtering.

The actual per-frame branch orchestration (calling YOLO/FELNet, cropping SSIs)
is Phase 9; this module is the association/bookkeeping core it drives, and is
tested on synthetic detections without needing images or models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kalman_filter import KalmanFilter
from .matching import (
    appearance_cost_matrix,
    gate_cost_matrix,
    iou_cost_matrix,
    iou_xywh,
    matching_cascade,
    min_cost_matching,
)
from .track import DEFAULT_MAX_AGE, Track


@dataclass
class Detection:
    """A single detection at a detection frame."""

    bbox_xywh: np.ndarray
    confidence: float = 1.0
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.bbox_xywh = np.asarray(self.bbox_xywh, dtype=np.float64)
        if self.embedding is not None:
            self.embedding = np.asarray(self.embedding, dtype=np.float64)


@dataclass
class TrackingOutput:
    """A tracking-branch localization result for one recovered target."""

    track_id: int
    bbox_xywh: np.ndarray
    confidence: float
    embedding: np.ndarray | None = None
    payload: Any = field(default=None)


def boundary_distance(bbox_xywh: np.ndarray, image_width: float, image_height: float) -> float:
    """Minimum distance from the box's edges to the image border (can be negative)."""

    x, y, w, h = bbox_xywh
    return float(min(x, y, image_width - (x + w), image_height - (y + h)))


def expansion_targets(
    candidate_tracks: list[Track],
    tau: int,
    d: float,
    image_size: tuple[float, float],
) -> list[Track]:
    """Filter unmatched-confirmed tracklets to the missed-detection expansion set.

    Paper A.6: at non-detection frames, also localize confirmed tracklets from
    the last detection cycle's unmatched set that are (1) confirmed, (2) not
    older than tau+1 since their last update, and (3) far enough (> d) from the
    image boundary that they probably have not left the frame.
    """

    width, height = image_size
    selected: list[Track] = []
    for track in candidate_tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update > tau + 1:
            continue
        if boundary_distance(track.bbox_xywh, width, height) <= d:
            continue
        selected.append(track)
    return selected


def filter_tracking_outputs(
    outputs: list[TrackingOutput],
    reference_boxes: np.ndarray,
    c1: float,
    i1: float,
    i2: float,
) -> list[TrackingOutput]:
    """Filter speculative tracking-branch outputs from recovered trajectories.

    Paper A.6 output-filtering rules:
      1. discard if confidence < c1;
      2. discard if IoU with any reference box (a current detection, or a
         confidently-tracked target) exceeds i1 -- a likely duplicate;
      3. among the survivors, if two overlap by more than i2, keep only the
         higher-confidence one.
    """

    reference_boxes = np.asarray(reference_boxes, dtype=np.float64).reshape(-1, 4)

    survivors: list[TrackingOutput] = []
    for out in outputs:
        if out.confidence < c1:
            continue
        if reference_boxes.shape[0] and np.any(iou_xywh(out.bbox_xywh, reference_boxes) > i1):
            continue
        survivors.append(out)

    # Dedup by descending confidence so the kept output of any overlapping pair
    # is always the more confident one.
    survivors.sort(key=lambda o: o.confidence, reverse=True)
    kept: list[TrackingOutput] = []
    for out in survivors:
        if any(iou_xywh(out.bbox_xywh, kept_out.bbox_xywh.reshape(1, 4))[0] > i2 for kept_out in kept):
            continue
        kept.append(out)
    return kept


class Tracker:
    """Trajectory set with the paper's modified-DeepSORT association.

    Hyperparameter defaults follow A.8 (tau=3 -> confirm after N=tau+1=4,
    100-miss deletion, d=5, c1/i1/i2=0.9/0.1/0.1). ``max_cosine_distance`` is
    not specified by the paper; the default is calibrated to FELNet's observed
    embedding distribution (Phase 4 eval: same-identity cosine distance ~0.24,
    different-identity ~0.89) and is expected to be tuned in Phase 10.
    """

    def __init__(
        self,
        tau: int = 3,
        max_cosine_distance: float = 0.5,
        max_iou_distance: float = 0.7,
        confirm_hits: int | None = None,
        max_age: int = DEFAULT_MAX_AGE,
        d: float = 5.0,
        c1: float = 0.9,
        i1: float = 0.1,
        i2: float = 0.1,
        kalman_filter: KalmanFilter | None = None,
    ) -> None:
        self.tau = tau
        self.confirm_hits = confirm_hits if confirm_hits is not None else tau + 1
        self.max_age = max_age
        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance = max_iou_distance
        self.d = d
        self.c1 = c1
        self.i1 = i1
        self.i2 = i2
        self.kf = kalman_filter or KalmanFilter()

        self.tracks: list[Track] = []
        self._next_id = 1
        # 𝒯_um' from the last detection frame: unmatched-but-still-confirmed
        # tracklets, the candidate pool for the tracking expansion set.
        self.last_unmatched_confirmed: list[Track] = []

    def predict(self) -> None:
        """Kalman-predict every live track one frame forward."""

        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections: list[Detection]) -> tuple[list, list, list]:
        """Run the detection-frame association (A.6 steps 1-7).

        Returns (matches, unmatched_track_indices, unmatched_detection_indices)
        where matches are (track_index, detection_index) pairs, indexing into
        ``self.tracks`` *before* deleted tracks are pruned.
        """

        matches, unmatched_tracks, unmatched_dets = self._match(detections)

        for ti, di in matches:
            self.tracks[ti].update(self.kf, detections[di].bbox_xywh, detections[di].embedding)
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()
        for di in unmatched_dets:
            self._initiate_track(detections[di])

        # Record 𝒯_um': tracks that stayed confirmed after going unmatched
        # (deleted ones dropped out by mark_missed) -- the expansion candidates.
        self.last_unmatched_confirmed = [
            self.tracks[ti] for ti in unmatched_tracks if self.tracks[ti].is_confirmed()
        ]

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        return matches, unmatched_tracks, unmatched_dets

    def _match(self, detections: list[Detection]) -> tuple[list, list, list]:
        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        detection_indices = list(range(len(detections)))

        # Stage 1: appearance cascade on confirmed tracks, spatially gated.
        if confirmed and detection_indices:
            cost = appearance_cost_matrix(self.tracks, detections, confirmed, detection_indices)
            gate_cost_matrix(cost, self.kf, self.tracks, detections, confirmed, detection_indices)
            matches_a, unmatched_a, unmatched_dets = matching_cascade(
                cost, self.max_cosine_distance, self.tracks, confirmed, detection_indices, self.max_age
            )
        else:
            matches_a, unmatched_a, unmatched_dets = [], list(confirmed), detection_indices

        # Stage 2: IoU assignment. Candidates = unconfirmed tracks + cascade-
        # unmatched confirmed tracks recent enough to still be trustworthy
        # (time_since_update <= tau+1, the paper's relaxation of DeepSORT's ==1).
        iou_candidates = unconfirmed + [
            k for k in unmatched_a if self.tracks[k].time_since_update <= self.tau + 1
        ]
        unmatched_a_stale = [k for k in unmatched_a if self.tracks[k].time_since_update > self.tau + 1]

        if iou_candidates and unmatched_dets:
            iou_cost = iou_cost_matrix(self.tracks, detections, iou_candidates, unmatched_dets)
            matches_b, unmatched_b, unmatched_dets = min_cost_matching(
                iou_cost, self.max_iou_distance, iou_candidates, unmatched_dets
            )
        else:
            matches_b, unmatched_b, unmatched_dets = [], list(iou_candidates), unmatched_dets

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_a_stale + unmatched_b))
        return matches, unmatched_tracks, unmatched_dets

    def _initiate_track(self, detection: Detection) -> None:
        track = Track.initiate(
            self.kf,
            self._next_id,
            detection.bbox_xywh,
            confirm_hits=self.confirm_hits,
            max_age=self.max_age,
            embedding=detection.embedding,
        )
        self.tracks.append(track)
        self._next_id += 1

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.is_confirmed()]

    def tracking_expansion_set(self, image_size: tuple[float, float]) -> list[Track]:
        """Missed-detection expansion set from the last detection frame's 𝒯_um'."""

        return expansion_targets(self.last_unmatched_confirmed, self.tau, self.d, image_size)
