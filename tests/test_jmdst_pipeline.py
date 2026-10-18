from __future__ import annotations

import unittest

import numpy as np

from jmdst.pipeline import JMDSTTracker


def _emb(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    return v / np.linalg.norm(v)


class _DummyImage:
    def __init__(self, size=(1000, 1000)):
        self.size = size


class MockDetector:
    """Returns preset detections per detection-frame call, in call order."""

    def __init__(self, per_call: list[list[tuple[np.ndarray, float]]]):
        self.per_call = per_call
        self.calls = 0

    def __call__(self, image):
        result = self.per_call[min(self.calls, len(self.per_call) - 1)]
        self.calls += 1
        return result


class MockLocalizer:
    """Embedding is looked up by nearest preset anchor; refined box == input box.

    Simulates a perfect single-object tracker (returns the queried box) with a
    stable per-identity embedding keyed to whichever anchor centre is closest.
    """

    def __init__(self, identity_anchors: list[tuple[np.ndarray, np.ndarray]], felnet_conf: float = 0.99):
        # identity_anchors: list of (reference_point_xy, embedding)
        self.identity_anchors = identity_anchors
        self.felnet_conf = felnet_conf

    def __call__(self, image, boxes_xywh):
        results = []
        for box in boxes_xywh:
            box = np.asarray(box, dtype=np.float64)
            center = box[:2] + box[2:] / 2
            emb = min(self.identity_anchors, key=lambda ia: np.linalg.norm(ia[0] - center))[1]
            results.append((emb, box.copy(), self.felnet_conf))
        return results


class _CountingLocalizer(MockLocalizer):
    """MockLocalizer that records how many times it was invoked."""

    def __init__(self, identity_anchors, felnet_conf: float = 0.99):
        super().__init__(identity_anchors, felnet_conf)
        self.calls = 0

    def __call__(self, image, boxes_xywh):
        self.calls += 1
        return super().__call__(image, boxes_xywh)


class DualBranchOrchestrationTests(unittest.TestCase):
    def test_detection_and_tracking_frames_alternate(self) -> None:
        # tau=3: frames 0,3,6 detect; 1,2,4,5 track. A target sits still and is
        # detected at every detection frame and tracked in between; it should
        # confirm (N=4 consecutive present frames) and output a stable ID.
        box = np.array([100.0, 100.0, 20.0, 20.0])
        emb = _emb(1, 0)
        detector = MockDetector([[(box, 0.9)]])
        localizer = MockLocalizer([(box[:2] + box[2:] / 2, emb)])
        jmdst = JMDSTTracker(detector, localizer, tau=3)

        outputs_per_frame = []
        for _ in range(6):
            outs = jmdst.process_frame(_DummyImage(), (1000, 1000))
            outputs_per_frame.append(outs)

        # Frames 0,1,2 -> hits 1,2,3 (tentative, no output). Frame 3 -> hit 4,
        # confirmed and output. Detector was only called on detection frames.
        self.assertEqual(detector.calls, 2)  # frames 0 and 3
        self.assertEqual(outputs_per_frame[0], [])
        self.assertEqual(outputs_per_frame[1], [])
        self.assertEqual(outputs_per_frame[2], [])
        self.assertEqual(len(outputs_per_frame[3]), 1)
        confirmed_id = outputs_per_frame[3][0].track_id
        # Stays confirmed and keeps its ID on subsequent tracking frames.
        self.assertEqual(outputs_per_frame[4][0].track_id, confirmed_id)
        self.assertEqual(outputs_per_frame[5][0].track_id, confirmed_id)

    def test_tracking_branch_keeps_target_alive_between_detections(self) -> None:
        # Without the tracking branch updating tracks, a tentative track would
        # be deleted before it could confirm. This verifies the whole point of
        # the dual-branch design.
        box = np.array([100.0, 100.0, 20.0, 20.0])
        emb = _emb(1, 0)
        jmdst = JMDSTTracker(
            MockDetector([[(box, 0.9)]]),
            MockLocalizer([(box[:2] + box[2:] / 2, emb)]),
            tau=3,
        )
        for _ in range(4):
            jmdst.process_frame(_DummyImage(), (1000, 1000))
        # One live, confirmed track survived across detection+tracking frames.
        self.assertEqual(len(jmdst.tracker.tracks), 1)
        self.assertTrue(jmdst.tracker.tracks[0].is_confirmed())
        self.assertGreaterEqual(jmdst.tracker.tracks[0].hits, 4)

    def test_missed_detection_recovered_by_expansion(self) -> None:
        # A confirmed target that goes undetected at a detection frame should be
        # kept alive by the tracking-branch expansion set and keep its ID.
        box = np.array([400.0, 400.0, 30.0, 30.0])
        emb = _emb(0, 1)
        center = box[:2] + box[2:] / 2
        # Detected at the first two detection frames (0, 3) to confirm, then
        # NOT detected at frame 6 (empty detection).
        detector = MockDetector([[(box, 0.9)], [(box, 0.9)], []])
        jmdst = JMDSTTracker(detector, MockLocalizer([(center, emb)]), tau=3)

        for _ in range(6):
            jmdst.process_frame(_DummyImage(), (1000, 1000))
        confirmed_before = [o.track_id for o in jmdst.process_frame(_DummyImage(), (1000, 1000))]  # frame 6: detection, misses
        self.assertTrue(jmdst.tracker.confirmed_tracks())
        original_id = jmdst.tracker.confirmed_tracks()[0].track_id

        # Frame 7 (tracking): expansion set should relocalize and keep the ID.
        out7 = jmdst.process_frame(_DummyImage(), (1000, 1000))
        self.assertIn(original_id, [o.track_id for o in out7])

    def test_unavailable_detection_falls_back_to_tracking_branch(self) -> None:
        # None (no result, e.g. a lost network reply) is not [] (nothing seen):
        # the frame must run the tracking branch, so a confirmed target is
        # relocalized and updated *at that frame* and keeps its ID.
        box = np.array([400.0, 400.0, 30.0, 30.0])
        center = box[:2] + box[2:] / 2
        detector = MockDetector([[(box, 0.9)], [(box, 0.9)], None])
        localizer = _CountingLocalizer([(center, _emb(0, 1))])
        jmdst = JMDSTTracker(detector, localizer, tau=3)
        for _ in range(6):
            jmdst.process_frame(_DummyImage(), (1000, 1000))
        original_id = jmdst.tracker.confirmed_tracks()[0].track_id
        calls_before = localizer.calls

        out6 = jmdst.process_frame(_DummyImage(), (1000, 1000))  # detection frame, result unavailable

        self.assertEqual(localizer.calls, calls_before + 1)  # tracking branch ran
        self.assertEqual([o.track_id for o in out6], [original_id])
        self.assertEqual(jmdst.tracker.confirmed_tracks()[0].time_since_update, 0)
        self.assertEqual(jmdst.unavailable_detection_frames, 1)
        self.assertEqual(jmdst.frame_index, 7)  # schedule unchanged: next detection at 9

    def test_unavailable_detection_before_any_track_is_harmless(self) -> None:
        box = np.array([100.0, 100.0, 20.0, 20.0])
        detector = MockDetector([None, [(box, 0.9)]])
        jmdst = JMDSTTracker(detector, MockLocalizer([(box[:2] + box[2:] / 2, _emb(1, 0))]), tau=3)
        for _ in range(3):
            self.assertEqual(jmdst.process_frame(_DummyImage(), (1000, 1000)), [])
        self.assertEqual(jmdst.tracker.tracks, [])
        jmdst.process_frame(_DummyImage(), (1000, 1000))  # frame 3: real detections start tracks
        self.assertEqual(len(jmdst.tracker.tracks), 1)

    def test_skip_frame_coasts_tracks_without_moving_the_schedule(self) -> None:
        box = np.array([100.0, 100.0, 20.0, 20.0])
        detector = MockDetector([[(box, 0.9)]])
        localizer = _CountingLocalizer([(box[:2] + box[2:] / 2, _emb(1, 0))])
        jmdst = JMDSTTracker(detector, localizer, tau=3)
        jmdst.process_frame(_DummyImage(), (1000, 1000))  # detection frame 0
        self.assertEqual(jmdst.last_branch, "detect")
        track = jmdst.tracker.tracks[0]
        calls = (detector.calls, localizer.calls)

        jmdst.skip_frame()
        jmdst.skip_frame()

        self.assertEqual((detector.calls, localizer.calls), calls)  # no model ran
        self.assertEqual(track.time_since_update, 2)  # coasted two camera frames
        self.assertEqual((jmdst.frame_index, jmdst.skipped_frames), (1, 2))
        jmdst.process_frame(_DummyImage(), (1000, 1000))
        self.assertEqual(jmdst.last_branch, "track")  # processed frame 1 of the tau cycle
        self.assertEqual(track.time_since_update, 0)

    def test_boundary_target_not_in_expansion(self) -> None:
        # A confirmed target that goes undetected AND sits within d of the image
        # border should not be recovered (likely left the frame).
        box = np.array([2.0, 400.0, 30.0, 30.0])  # 2px from left edge < d=5
        emb = _emb(0, 1)
        center = box[:2] + box[2:] / 2
        detector = MockDetector([[(box, 0.9)], [(box, 0.9)], []])
        jmdst = JMDSTTracker(detector, MockLocalizer([(center, emb)]), tau=3, d=5)

        for _ in range(7):  # through frame 6 (detection with a miss)
            jmdst.process_frame(_DummyImage(), (1000, 1000))
        out7 = jmdst.process_frame(_DummyImage(), (1000, 1000))  # tracking frame
        # The boundary target is excluded from the expansion set -> no output.
        self.assertEqual(out7, [])

    def test_no_detections_first_frame_produces_no_tracks(self) -> None:
        jmdst = JMDSTTracker(MockDetector([[]]), MockLocalizer([]), tau=3)
        out = jmdst.process_frame(_DummyImage(), (1000, 1000))
        self.assertEqual(out, [])
        self.assertEqual(len(jmdst.tracker.tracks), 0)


class DetectionBranchEmbeddingTests(unittest.TestCase):
    """Embeddings are only computed when association will actually read them.

    appearance_cost_matrix is their sole consumer and IoU-only association
    skips it, so calling the localizer on the detection branch there is one
    wasted FELNet forward per detection per detection frame.
    """

    def _run_detection_frame(self, *, use_appearance: bool):
        box = np.array([100.0, 100.0, 20.0, 20.0])
        localizer = _CountingLocalizer([(np.array([110.0, 110.0]), _emb(1.0, 0.0))])
        jmdst = JMDSTTracker(
            MockDetector([[(box, 0.9)]]), localizer, tau=3, use_appearance=use_appearance
        )
        jmdst.process_frame(_DummyImage(), (1000, 1000))
        return jmdst, localizer

    def test_iou_only_skips_the_localizer_on_detection_frames(self) -> None:
        jmdst, localizer = self._run_detection_frame(use_appearance=False)
        self.assertEqual(localizer.calls, 0)
        # A track is still created -- only the unused embedding is skipped.
        self.assertEqual(len(jmdst.tracker.tracks), 1)
        self.assertIsNone(jmdst.tracker.tracks[0].embedding_history[-1]
                          if jmdst.tracker.tracks[0].embedding_history else None)

    def test_appearance_mode_still_computes_embeddings(self) -> None:
        jmdst, localizer = self._run_detection_frame(use_appearance=True)
        self.assertEqual(localizer.calls, 1)
        self.assertTrue(jmdst.tracker.tracks[0].embedding_history)

    def test_iou_only_still_uses_the_localizer_on_tracking_frames(self) -> None:
        # The tracking branch needs FELNet for box refinement regardless of
        # association mode -- the skip must not leak into it.
        box = np.array([100.0, 100.0, 20.0, 20.0])
        localizer = _CountingLocalizer([(np.array([110.0, 110.0]), _emb(1.0, 0.0))])
        jmdst = JMDSTTracker(
            MockDetector([[(box, 0.9)]]), localizer, tau=3, use_appearance=False
        )
        jmdst.process_frame(_DummyImage(), (1000, 1000))   # detection frame
        self.assertEqual(localizer.calls, 0)
        jmdst.process_frame(_DummyImage(), (1000, 1000))   # tracking frame
        self.assertEqual(localizer.calls, 1)


if __name__ == "__main__":
    unittest.main()
