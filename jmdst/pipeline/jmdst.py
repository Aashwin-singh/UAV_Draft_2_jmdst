"""JMDST application routine -- the dual-branch loop (paper Algorithm 1).

Orchestrates the alternation between the periodic detection branch and the
per-frame tracking branch, driving the Phase 8 Tracker. Model inference is
injected as two callables so the branch logic (the paper's contribution) is
testable without YOLO/FELNet:

    detector(image) -> list[(bbox_xywh, confidence)]
        multi-object detection results (YOLOv11 + NMS/conf post-processing).

    localizer(image, list[bbox_xywh])
        -> list[(embedding, refined_bbox_xywh, felnet_confidence)]
        For each input box, crop its SSI, run FELNet, select the anchor
        (paper Sec. 2.2 rule) using that box as the reference, and return the
        selected embedding, the decoded refined box (Eq. 2, mapped back to
        image coords), and the anchor's confidence. The detection branch uses
        only the embedding; the tracking branch uses all three.

See jmdst.pipeline.models for the real YOLO/FELNet-backed implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from jmdst.tracking import Detection, Tracker, TrackingOutput, filter_tracking_outputs

Detector = Callable[[object], Sequence[tuple[np.ndarray, float]]]
Localizer = Callable[[object, Sequence[np.ndarray]], Sequence[tuple[np.ndarray, np.ndarray, float]]]


@dataclass
class ConfirmedOutput:
    """One confirmed-state tracklet output at the current frame."""

    track_id: int
    bbox_xywh: np.ndarray
    confidence: float


class JMDSTTracker:
    """Runs the paper's dual-branch application routine over a frame sequence.

    A detection frame occurs when ``frame_index % tau == 0``; every other frame
    runs the tracking branch. At each frame, ``process_frame`` returns the
    confirmed-state tracklets updated at that frame (Algorithm 1's output).
    """

    def __init__(
        self,
        detector: Detector,
        localizer: Localizer,
        tau: int = 3,
        d: float = 5.0,
        c1: float = 0.9,
        i1: float = 0.1,
        i2: float = 0.1,
        max_cosine_distance: float = 0.5,
        max_age: int = 100,
    ) -> None:
        self.detector = detector
        self.localizer = localizer
        self.tau = tau
        self.tracker = Tracker(
            tau=tau, d=d, c1=c1, i1=i1, i2=i2,
            max_cosine_distance=max_cosine_distance, max_age=max_age,
        )
        self.frame_index = 0

    # -- public API ---------------------------------------------------------

    def process_frame(self, image: object, image_size: tuple[float, float]) -> list[ConfirmedOutput]:
        """Process one frame; return its confirmed-state tracklet outputs."""

        if self.frame_index % self.tau == 0:
            self._detection_frame(image)
        else:
            self._tracking_frame(image, image_size)
        self.frame_index += 1
        return self._confirmed_outputs()

    # -- branches -----------------------------------------------------------

    def _detection_frame(self, image: object) -> None:
        raw = list(self.detector(image))
        boxes = [np.asarray(box, dtype=np.float64) for box, _conf in raw]
        confidences = [float(conf) for _box, conf in raw]

        embeddings: list[np.ndarray | None]
        if boxes:
            localized = self.localizer(image, boxes)
            embeddings = [np.asarray(emb, dtype=np.float64) for emb, _rb, _fc in localized]
        else:
            embeddings = []

        detections = [
            Detection(boxes[i], confidences[i], embeddings[i]) for i in range(len(boxes))
        ]
        self.tracker.predict()
        self.tracker.update(detections)
        # Stamp the detection confidence on each matched track for output.
        for track in self.tracker.tracks:
            if track.time_since_update == 0:
                # Find the detection this track was updated with (nearest box).
                track.last_confidence = self._nearest_detection_confidence(track, detections)

    def _tracking_frame(self, image: object, image_size: tuple[float, float]) -> None:
        self.tracker.predict()

        # Split the live tracks into the base tracking set (matched/created at
        # the last detection frame) and the missed-detection expansion set.
        unmatched_ids = {t.track_id for t in self.tracker.last_unmatched_confirmed}
        base = [t for t in self.tracker.tracks if t.track_id not in unmatched_ids]
        expansion = self.tracker.tracking_expansion_set(image_size)

        to_localize = base + expansion
        if to_localize:
            localized = list(self.localizer(image, [t.bbox_xywh for t in to_localize]))
        else:
            localized = []
        base_results = localized[: len(base)]
        expansion_results = localized[len(base):]

        updated_ids: set[int] = set()
        base_boxes: list[np.ndarray] = []

        # Base targets are trusted -- always updated with their localization.
        for track, (emb, refined_box, felnet_conf) in zip(base, base_results):
            refined = np.asarray(refined_box, dtype=np.float64)
            track.update(self.tracker.kf, refined, np.asarray(emb, dtype=np.float64))
            track.last_confidence = float(felnet_conf)
            updated_ids.add(track.track_id)
            base_boxes.append(refined)

        # Expansion (recovered) outputs are speculative -> filtered before use.
        exp_outputs = [
            TrackingOutput(track.track_id, np.asarray(rb, dtype=np.float64), float(fc), np.asarray(emb, dtype=np.float64))
            for track, (emb, rb, fc) in zip(expansion, expansion_results)
        ]
        reference = np.array(base_boxes, dtype=np.float64) if base_boxes else np.zeros((0, 4))
        kept = {o.track_id: o for o in filter_tracking_outputs(exp_outputs, reference, self.tracker.c1, self.tracker.i1, self.tracker.i2)}

        expansion_by_id = {t.track_id: t for t in expansion}
        for track_id, output in kept.items():
            track = expansion_by_id[track_id]
            track.update(self.tracker.kf, output.bbox_xywh, output.embedding)
            track.last_confidence = float(output.confidence)
            updated_ids.add(track_id)

        # Everything not localized/kept this frame counts as a miss.
        for track in self.tracker.tracks:
            if track.track_id not in updated_ids:
                track.mark_missed()
        self.tracker.tracks = [t for t in self.tracker.tracks if not t.is_deleted()]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _nearest_detection_confidence(track, detections: list[Detection]) -> float:
        if not detections:
            return 1.0
        track_box = track.bbox_xywh
        best_conf, best_dist = 1.0, float("inf")
        for det in detections:
            dist = float(np.linalg.norm(track_box[:2] - det.bbox_xywh[:2]))
            if dist < best_dist:
                best_dist, best_conf = dist, det.confidence
        return best_conf

    def _confirmed_outputs(self) -> list[ConfirmedOutput]:
        outputs = []
        for track in self.tracker.tracks:
            # Algorithm 1 outputs confirmed tracklets updated at this frame.
            if not (track.is_confirmed() and track.time_since_update == 0):
                continue
            box = track.bbox_xywh
            # The Kalman box can be transiently degenerate for tiny/shrinking
            # targets (a predict-step overshoot in height/aspect that recovers
            # the next frame). Skip such frames rather than emit an invalid box;
            # the track lives on internally.
            if not np.all(np.isfinite(box)) or box[2] <= 0 or box[3] <= 0:
                continue
            outputs.append(
                ConfirmedOutput(
                    track_id=track.track_id,
                    bbox_xywh=box,
                    confidence=float(getattr(track, "last_confidence", 1.0)),
                )
            )
        return outputs
