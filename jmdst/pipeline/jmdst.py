"""JMDST application routine -- the dual-branch loop (paper Algorithm 1).

Orchestrates the alternation between the periodic detection branch and the
per-frame tracking branch, driving the Phase 8 Tracker. Model inference is
injected as two callables so the branch logic (the paper's contribution) is
testable without YOLO/FELNet:

    detector(image) -> list[(bbox_xywh, confidence)] | None
        multi-object detection results (YOLOv11 + NMS/conf post-processing).
        ``None`` means *no result is available* for this frame -- e.g. a
        remote detector whose reply was lost or missed its deadline -- and is
        distinct from ``[]`` ("looked, found nothing"). On ``None`` the frame
        runs the tracking branch instead, the paper's own mechanism for
        carrying targets through a missed detection.

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

Detector = Callable[[object], Sequence[tuple[np.ndarray, float]] | None]
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
        max_cosine_distance: float = 0.3,
        max_age: int = 100,
        use_appearance: bool = True,
        use_nsa: bool = False,
        nsa_tracking_branch: bool = False,
        cmc: object | None = None,
        use_byte: bool = False,
        byte_conf_high: float = 0.55,
        byte_iou_distance: float = 0.5,
    ) -> None:
        # With BYTE on, the detector is configured to emit boxes down to the
        # LOW threshold and this class splits them: >= byte_conf_high behaves
        # exactly as before, the rest becomes the weak pool for stage 3.
        self.use_byte = use_byte
        self.byte_conf_high = byte_conf_high
        # NSA on the tracking branch is a separate switch because the signal it
        # would consume is different in kind. At detection frames the
        # confidence is YOLO's calibrated detector score; on tracking frames it
        # is FELNet's *anchor* confidence, which answers "does this anchor
        # cover a target", not "is this box accurate". Measured 2026-08-10:
        # feeding the latter into NSA costs ~1.6 HOTA on both datasets.
        self.nsa_tracking_branch = nsa_tracking_branch
        self.detector = detector
        self.localizer = localizer
        self.tau = tau
        # Optional CameraMotionCompensator. When present, every frame's global
        # warp is applied to all track states before predict/associate, so
        # Kalman state and incoming detections share a coordinate frame.
        self.cmc = cmc
        self.tracker = Tracker(
            tau=tau, d=d, c1=c1, i1=i1, i2=i2,
            max_cosine_distance=max_cosine_distance, max_age=max_age,
            use_appearance=use_appearance, use_nsa=use_nsa,
            use_byte=use_byte, byte_iou_distance=byte_iou_distance,
        )
        self.frame_index = 0
        self.unavailable_detection_frames = 0  # detector returned None
        self.skipped_frames = 0  # camera frames the caller could not process
        self.last_branch: str | None = None  # "detect" | "track" | "fallback"

    # -- public API ---------------------------------------------------------

    def skip_frame(self) -> None:
        """A camera frame passed that will not be processed (real-time overload).

        Coasts every track one frame on the Kalman motion model, so motion and
        uncertainty stay in step with camera time across the gap. Nothing else
        runs and nothing is output. ``frame_index`` -- and so the tau schedule
        -- counts *processed* frames only: scheduling on camera frames would
        let a steadily overloaded edge alias onto, or entirely past, the
        detection frames.
        """

        self.tracker.predict()
        self.skipped_frames += 1

    def process_frame(self, image: object, image_size: tuple[float, float]) -> list[ConfirmedOutput]:
        """Process one frame; return its confirmed-state tracklet outputs."""

        # Camera motion compensation runs first, on every frame, so that both
        # branches see track states already aligned to this frame. Must precede
        # tracker.predict(): CMC changes the coordinate frame, predict advances
        # motion within it.
        if self.cmc is not None:
            warp = self.cmc.estimate(image)
            if warp is not None:
                self.cmc.apply(self.tracker.tracks, warp)

        if self.frame_index % self.tau == 0:
            raw = self.detector(image)
            if raw is None:
                # No detection result for this frame (e.g. a remote detector's
                # reply was lost or late). Fall back to the tracking branch so
                # Kalman + FELNet carry every target through the gap; the next
                # detection frame stays on the fixed tau schedule.
                self.unavailable_detection_frames += 1
                self.last_branch = "fallback"
                self._tracking_frame(image, image_size)
            else:
                self.last_branch = "detect"
                self._detection_frame(image, list(raw))
        else:
            self.last_branch = "track"
            self._tracking_frame(image, image_size)
        self.frame_index += 1
        return self._confirmed_outputs()

    # -- branches -----------------------------------------------------------

    def _detection_frame(self, image: object, raw: list[tuple[np.ndarray, float]]) -> None:

        # BYTE: split the detector's output into the strong pool (business as
        # usual) and the weak pool (stage-3 rescue only). Without BYTE the
        # detector is already thresholded and everything is strong.
        low_detections: list[Detection] = []
        if self.use_byte:
            weak = [(b, c) for b, c in raw if float(c) < self.byte_conf_high]
            raw = [(b, c) for b, c in raw if float(c) >= self.byte_conf_high]
            # Weak boxes are matched IoU-only by definition, so they never need
            # an embedding -- no FELNet cost is added by widening the threshold.
            low_detections = [
                Detection(np.asarray(b, dtype=np.float64), float(c), None) for b, c in weak
            ]

        boxes = [np.asarray(box, dtype=np.float64) for box, _conf in raw]
        confidences = [float(conf) for _box, conf in raw]

        embeddings: list[np.ndarray | None]
        if boxes and self.tracker.use_appearance:
            localized = self.localizer(image, boxes)
            embeddings = [np.asarray(emb, dtype=np.float64) for emb, _rb, _fc in localized]
        else:
            # Under IoU-only association nothing reads these. appearance_cost_matrix
            # is their sole consumer and _match skips it entirely, so running
            # FELNet here would be one forward pass per detection per detection
            # frame for a value that is stored and never looked at.
            embeddings = [None] * len(boxes)

        detections = [
            Detection(boxes[i], confidences[i], embeddings[i]) for i in range(len(boxes))
        ]
        self.tracker.predict()
        self.tracker.update(detections, low_detections)
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
            # NSA uses FELNet's anchor confidence here -- the tracking branch's
            # own measure of how much to trust this localization.
            track.update(
                self.tracker.kf, refined, np.asarray(emb, dtype=np.float64),
                float(felnet_conf) if self.nsa_tracking_branch else None,
            )
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
            track.update(
                self.tracker.kf, output.bbox_xywh, output.embedding,
                float(output.confidence) if self.nsa_tracking_branch else None,
            )
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
