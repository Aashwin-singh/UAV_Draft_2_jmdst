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


if __name__ == "__main__":
    unittest.main()
