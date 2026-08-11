from __future__ import annotations

import unittest

import numpy as np

from jmdst.tracking import (
    Detection,
    Tracker,
    TrackingOutput,
    boundary_distance,
    expansion_targets,
    filter_tracking_outputs,
)
from jmdst.tracking.kalman_filter import KalmanFilter
from jmdst.tracking.matching import (
    appearance_cost_matrix,
    iou_cost_matrix,
    iou_xywh,
    matching_cascade,
    min_cost_matching,
)
from jmdst.tracking.track import Track


def _emb(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    return v / np.linalg.norm(v)


class IoUTests(unittest.TestCase):
    def test_identical_boxes_iou_one(self) -> None:
        box = np.array([10.0, 10.0, 20.0, 20.0])
        self.assertAlmostEqual(float(iou_xywh(box, box[None, :])[0]), 1.0)

    def test_disjoint_boxes_iou_zero(self) -> None:
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([100.0, 100.0, 10.0, 10.0])
        self.assertAlmostEqual(float(iou_xywh(a, b[None, :])[0]), 0.0)

    def test_half_overlap(self) -> None:
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([5.0, 0.0, 10.0, 10.0])  # overlap 50, union 150
        self.assertAlmostEqual(float(iou_xywh(a, b[None, :])[0]), 50.0 / 150.0)


class MinCostMatchingTests(unittest.TestCase):
    def test_matches_within_threshold_only(self) -> None:
        # 2 tracks x 2 detections; the cheap pairing is (0,1) and (1,0).
        cost = np.array([[0.9, 0.1], [0.2, 0.8]])
        matches, ut, ud = min_cost_matching(cost, 0.5, [0, 1], [0, 1])
        self.assertEqual(sorted(matches), [(0, 1), (1, 0)])
        self.assertEqual(ut, [])
        self.assertEqual(ud, [])

    def test_rejects_pairs_over_max_distance(self) -> None:
        cost = np.array([[0.9]])  # only pairing is too costly
        matches, ut, ud = min_cost_matching(cost, 0.5, [0], [0])
        self.assertEqual(matches, [])
        self.assertEqual(ut, [0])
        self.assertEqual(ud, [0])

    def test_empty_inputs(self) -> None:
        cost = np.zeros((0, 0))
        self.assertEqual(min_cost_matching(cost, 0.5, [], []), ([], [], []))


class CostMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kf = KalmanFilter()

    def _track(self, track_id: int, bbox, embedding=None) -> Track:
        return Track.initiate(self.kf, track_id, np.array(bbox, dtype=np.float64), embedding=embedding)

    def test_appearance_cost_is_cosine_distance(self) -> None:
        tracks = [self._track(1, [0, 0, 10, 10], embedding=_emb(1, 0))]
        detections = [
            Detection(np.array([0, 0, 10, 10]), embedding=_emb(1, 0)),  # identical -> dist 0
            Detection(np.array([0, 0, 10, 10]), embedding=_emb(0, 1)),  # orthogonal -> dist 1
        ]
        cost = appearance_cost_matrix(tracks, detections, [0], [0, 1])
        self.assertAlmostEqual(cost[0, 0], 0.0, places=6)
        self.assertAlmostEqual(cost[0, 1], 1.0, places=6)

    def test_missing_embedding_gets_max_cost(self) -> None:
        tracks = [self._track(1, [0, 0, 10, 10], embedding=None)]
        detections = [Detection(np.array([0, 0, 10, 10]), embedding=_emb(1, 0))]
        cost = appearance_cost_matrix(tracks, detections, [0], [0])
        self.assertEqual(cost[0, 0], 1.0)

    def test_iou_cost_matrix(self) -> None:
        tracks = [self._track(1, [0, 0, 10, 10])]
        detections = [
            Detection(np.array([0, 0, 10, 10])),  # IoU 1 -> cost 0
            Detection(np.array([100, 100, 10, 10])),  # IoU 0 -> cost 1
        ]
        cost = iou_cost_matrix(tracks, detections, [0], [0, 1])
        self.assertAlmostEqual(cost[0, 0], 0.0)
        self.assertAlmostEqual(cost[0, 1], 1.0)


class CascadePriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kf = KalmanFilter()

    def test_recent_track_wins_contested_detection(self) -> None:
        # Two confirmed tracks with identical embeddings both want detection 0.
        # The cascade must award it to the more recently updated track (age 1).
        recent = Track.initiate(self.kf, 1, np.array([0.0, 0, 10, 10]), embedding=_emb(1, 0))
        stale = Track.initiate(self.kf, 2, np.array([0.0, 0, 10, 10]), embedding=_emb(1, 0))
        recent.time_since_update = 1
        stale.time_since_update = 3
        tracks = [recent, stale]
        detections = [Detection(np.array([0.0, 0, 10, 10]), embedding=_emb(1, 0))]

        cost = appearance_cost_matrix(tracks, detections, [0, 1], [0])
        matches, unmatched_tracks, unmatched_dets = matching_cascade(
            cost, 0.5, tracks, [0, 1], [0], cascade_depth=10
        )
        self.assertEqual(matches, [(0, 0)])  # recent track (index 0) wins
        self.assertEqual(unmatched_tracks, [1])
        self.assertEqual(unmatched_dets, [])


class TrackerAssociationTests(unittest.TestCase):
    def _run_detection_frame(self, tracker: Tracker, detections: list[Detection]) -> None:
        tracker.predict()
        tracker.update(detections)

    def test_new_detections_create_tentative_tracks(self) -> None:
        tracker = Tracker()
        tracker.update([Detection(np.array([0, 0, 10, 10]), embedding=_emb(1, 0))])
        self.assertEqual(len(tracker.tracks), 1)
        self.assertTrue(tracker.tracks[0].is_tentative())

    def test_track_confirms_after_tau_plus_one_detections(self) -> None:
        tracker = Tracker(tau=3)  # confirm_hits = 4
        box = np.array([0.0, 0, 10, 10])
        emb = _emb(1, 0)
        tracker.update([Detection(box, embedding=emb)])  # hit 1
        for _ in range(2):  # hits 2, 3
            self._run_detection_frame(tracker, [Detection(box, embedding=emb)])
            self.assertTrue(tracker.tracks[0].is_tentative())
        self._run_detection_frame(tracker, [Detection(box, embedding=emb)])  # hit 4
        self.assertTrue(tracker.tracks[0].is_confirmed())

    def test_large_jump_beyond_kalman_gate_is_not_matched(self) -> None:
        # A confirmed track whose detection reappears displaced far beyond the
        # Kalman motion gate is NOT rescued by appearance -- DeepSORT gates all
        # appearance matches by Mahalanobis distance. This is the paper's own
        # documented failure mode (Sec. 3.5.4: rapid target motion beyond the
        # Kalman prediction range causes ID switches).
        tracker = Tracker(tau=1)  # confirm after 2 hits
        emb = _emb(1, 0, 0)
        tracker.update([Detection(np.array([0.0, 0, 20, 20]), embedding=emb)])
        self._run_detection_frame(tracker, [Detection(np.array([0.0, 0, 20, 20]), embedding=emb)])
        self.assertTrue(tracker.tracks[0].is_confirmed())

        # 60px jump for a 20px box: gating distance ~825 >> 9.49 threshold.
        self._run_detection_frame(tracker, [Detection(np.array([60.0, 0, 20, 20]), embedding=emb)])
        # Old confirmed track went unmatched (survives, it's confirmed); the
        # jumped detection started a new track -> an ID switch, not a match.
        self.assertEqual(len(tracker.tracks), 2)

    def test_appearance_disambiguates_crossed_nearby_detections(self) -> None:
        # Two confirmed, nearby tracks with distinct appearances. The two new
        # detections keep their embeddings but sit at positions that, by IoU
        # alone, would swap the identities. Appearance matching (run first, in
        # the cascade) must preserve identity instead. Both cross-pairings are
        # within the Kalman gate (~5px), so position doesn't veto them.
        tracker = Tracker(tau=1)  # confirm after 2 hits
        emb_a, emb_b = _emb(1, 0), _emb(0, 1)
        box_a, box_b = np.array([0.0, 0, 20, 20]), np.array([6.0, 0, 20, 20])

        for _ in range(2):  # confirm both A (id 1) and B (id 2)
            self._run_detection_frame(tracker, [Detection(box_a, embedding=emb_a), Detection(box_b, embedding=emb_b)])
        ids = {t.track_id for t in tracker.tracks}
        self.assertEqual(ids, {1, 2})

        # det at x=5 carries A's embedding; det at x=1 carries B's embedding.
        # Pure IoU would pair A(pred 0)->det(x=1) and B(pred 6)->det(x=5).
        self._run_detection_frame(
            tracker,
            [Detection(np.array([1.0, 0, 20, 20]), embedding=emb_b),
             Detection(np.array([5.0, 0, 20, 20]), embedding=emb_a)],
        )
        self.assertEqual(len(tracker.tracks), 2)
        track_a = next(t for t in tracker.tracks if t.track_id == 1)
        # Track A followed its embedding to x=5, not IoU to x=1.
        self.assertGreater(track_a.bbox_xywh[0], 2.5)

    def test_unconfirmed_track_deleted_on_miss(self) -> None:
        tracker = Tracker(tau=3)
        tracker.update([Detection(np.array([0, 0, 10, 10]), embedding=_emb(1, 0))])
        # Next detection frame: a totally different, far-away target.
        self._run_detection_frame(tracker, [Detection(np.array([500, 500, 10, 10]), embedding=_emb(0, 1))])
        # The original tentative track went unmatched -> deleted immediately;
        # only the new tentative track remains.
        self.assertEqual(len(tracker.tracks), 1)
        np.testing.assert_allclose(tracker.tracks[0].bbox_xywh, [500, 500, 10, 10], atol=1e-6)

    def test_distinct_targets_get_distinct_ids(self) -> None:
        tracker = Tracker(tau=1)
        a = Detection(np.array([0.0, 0, 10, 10]), embedding=_emb(1, 0))
        b = Detection(np.array([200.0, 200, 10, 10]), embedding=_emb(0, 1))
        tracker.update([a, b])
        self.assertEqual({t.track_id for t in tracker.tracks}, {1, 2})


class AppearanceAblationTests(unittest.TestCase):
    """use_appearance=False must fall back to IoU-only association (Sec. 3.4.2 ablation)."""

    def _run(self, tracker: Tracker, detections: list[Detection]) -> None:
        tracker.predict()
        tracker.update(detections)

    def test_iou_only_swaps_identities_that_appearance_preserves(self) -> None:
        # Two confirmed nearby tracks with distinct appearances; the new
        # detections keep their embeddings but sit at positions that IoU alone
        # would pair the other way. With appearance ON, identity is preserved
        # (verified in TrackerAssociationTests); with appearance OFF, the
        # tracker must follow IoU instead -- proving the flag really changes
        # the association path rather than being ignored.
        emb_a, emb_b = _emb(1, 0), _emb(0, 1)
        box_a, box_b = np.array([0.0, 0, 20, 20]), np.array([6.0, 0, 20, 20])

        tracker = Tracker(tau=1, use_appearance=False)
        for _ in range(2):
            self._run(tracker, [Detection(box_a, embedding=emb_a), Detection(box_b, embedding=emb_b)])
        self.assertEqual({t.track_id for t in tracker.tracks}, {1, 2})

        # det at x=1 carries B's embedding, det at x=5 carries A's embedding.
        self._run(
            tracker,
            [Detection(np.array([1.0, 0, 20, 20]), embedding=emb_b),
             Detection(np.array([5.0, 0, 20, 20]), embedding=emb_a)],
        )
        track_a = next(t for t in tracker.tracks if t.track_id == 1)
        # IoU-only: track A (predicted near x=0) takes the x=1 detection,
        # ignoring that it carries B's embedding.
        self.assertLess(track_a.bbox_xywh[0], 2.5)

    def test_iou_only_still_tracks_a_simple_target(self) -> None:
        # Sanity: the ablation must remain a working tracker, not a broken one.
        tracker = Tracker(tau=1, use_appearance=False)
        box = np.array([10.0, 10, 20, 20])
        for _ in range(3):
            self._run(tracker, [Detection(box, embedding=_emb(1, 0))])
        self.assertEqual(len(tracker.tracks), 1)
        self.assertTrue(tracker.tracks[0].is_confirmed())


class ExpansionTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kf = KalmanFilter()

    def _confirmed_track(self, bbox, time_since_update=1) -> Track:
        track = Track.initiate(self.kf, 1, np.array(bbox, dtype=np.float64), confirm_hits=1)
        track.update(self.kf, np.array(bbox, dtype=np.float64))  # -> confirmed
        track.time_since_update = time_since_update
        return track

    def test_boundary_distance(self) -> None:
        # Box [10,10,20,20] in a 100x100 image: nearest edge is left/top at 10.
        self.assertAlmostEqual(boundary_distance(np.array([10, 10, 20, 20]), 100, 100), 10.0)
        # Box hanging off the right edge -> negative distance.
        self.assertAlmostEqual(boundary_distance(np.array([90, 10, 20, 20]), 100, 100), -10.0)

    def test_confirmed_recent_interior_track_selected(self) -> None:
        track = self._confirmed_track([40, 40, 20, 20], time_since_update=2)
        selected = expansion_targets([track], tau=3, d=5, image_size=(100, 100))
        self.assertEqual(selected, [track])

    def test_stale_track_excluded(self) -> None:
        track = self._confirmed_track([40, 40, 20, 20], time_since_update=5)  # > tau+1=4
        self.assertEqual(expansion_targets([track], tau=3, d=5, image_size=(100, 100)), [])

    def test_boundary_track_excluded(self) -> None:
        track = self._confirmed_track([2, 40, 20, 20], time_since_update=1)  # 2px from left edge < d=5
        self.assertEqual(expansion_targets([track], tau=3, d=5, image_size=(100, 100)), [])

    def test_unconfirmed_track_excluded(self) -> None:
        track = Track.initiate(self.kf, 1, np.array([40.0, 40, 20, 20]), confirm_hits=4)
        self.assertTrue(track.is_tentative())
        self.assertEqual(expansion_targets([track], tau=3, d=5, image_size=(100, 100)), [])


class OutputFilteringTests(unittest.TestCase):
    def test_low_confidence_discarded(self) -> None:
        outputs = [TrackingOutput(1, np.array([0.0, 0, 10, 10]), confidence=0.5)]
        kept = filter_tracking_outputs(outputs, np.zeros((0, 4)), c1=0.9, i1=0.1, i2=0.1)
        self.assertEqual(kept, [])

    def test_overlapping_detection_discarded(self) -> None:
        # A high-confidence output that overlaps an existing detection box is a
        # duplicate -> discarded.
        outputs = [TrackingOutput(1, np.array([0.0, 0, 10, 10]), confidence=0.95)]
        reference = np.array([[1.0, 1, 10, 10]])  # IoU well above i1
        kept = filter_tracking_outputs(outputs, reference, c1=0.9, i1=0.1, i2=0.1)
        self.assertEqual(kept, [])

    def test_isolated_confident_output_kept(self) -> None:
        outputs = [TrackingOutput(1, np.array([0.0, 0, 10, 10]), confidence=0.95)]
        reference = np.array([[500.0, 500, 10, 10]])  # far away
        kept = filter_tracking_outputs(outputs, reference, c1=0.9, i1=0.1, i2=0.1)
        self.assertEqual(len(kept), 1)

    def test_mutual_overlap_keeps_higher_confidence(self) -> None:
        outputs = [
            TrackingOutput(1, np.array([0.0, 0, 10, 10]), confidence=0.92),
            TrackingOutput(2, np.array([1.0, 1, 10, 10]), confidence=0.97),  # overlaps #1
            TrackingOutput(3, np.array([500.0, 500, 10, 10]), confidence=0.93),  # isolated
        ]
        kept = filter_tracking_outputs(outputs, np.zeros((0, 4)), c1=0.9, i1=0.1, i2=0.1)
        kept_ids = {o.track_id for o in kept}
        self.assertEqual(kept_ids, {2, 3})  # #1 dropped in favor of higher-conf #2


class ByteAssociationTests(unittest.TestCase):
    """ByteTrack's second stage: weak boxes continue tracks, never start them."""

    BOX = np.array([100.0, 100.0, 40.0, 40.0])

    def _confirmed_tracker(self, **kwargs) -> Tracker:
        """A tracker holding one confirmed track on BOX."""

        tracker = Tracker(tau=3, use_appearance=False, **kwargs)
        for _ in range(5):  # N = tau + 1 = 4 to confirm, plus one
            tracker.predict()
            tracker.update([Detection(self.BOX.copy(), 0.9)])
        self.assertTrue(tracker.tracks and tracker.tracks[0].is_confirmed())
        return tracker

    def test_low_detections_never_create_tracks(self) -> None:
        # The whole safety property: a frame of nothing but weak boxes, far
        # from any track, must not spawn anything.
        tracker = Tracker(tau=3, use_appearance=False, use_byte=True)
        tracker.predict()
        tracker.update([], [Detection(np.array([10.0, 10.0, 30.0, 30.0]), 0.2)])
        self.assertEqual(len(tracker.tracks), 0)

    def test_low_detection_keeps_a_confirmed_track_alive(self) -> None:
        tracker = self._confirmed_tracker(use_byte=True)
        track = tracker.tracks[0]
        tracker.predict()
        tracker.update([], [Detection(self.BOX.copy(), 0.2)])

        self.assertEqual(track.time_since_update, 0)
        self.assertTrue(track.is_confirmed())

    def test_without_byte_the_same_frame_is_a_miss(self) -> None:
        tracker = self._confirmed_tracker(use_byte=False)
        track = tracker.tracks[0]
        tracker.predict()
        tracker.update([], [Detection(self.BOX.copy(), 0.2)])

        # low_detections must be ignored entirely when BYTE is off.
        self.assertEqual(track.time_since_update, 1)

    def test_spatially_distant_low_detection_is_rejected(self) -> None:
        tracker = self._confirmed_tracker(use_byte=True)
        track = tracker.tracks[0]
        tracker.predict()
        tracker.update([], [Detection(np.array([600.0, 600.0, 40.0, 40.0]), 0.2)])

        self.assertEqual(track.time_since_update, 1)

    def test_tentative_tracks_are_not_rescued(self) -> None:
        # Letting weak evidence carry a tentative track toward confirmation
        # would undermine the paper's N = tau+1 rule.
        tracker = Tracker(tau=3, use_appearance=False, use_byte=True)
        tracker.predict()
        tracker.update([Detection(self.BOX.copy(), 0.9)])
        track = tracker.tracks[0]
        self.assertTrue(track.is_tentative())

        tracker.predict()
        tracker.update([], [Detection(self.BOX.copy(), 0.2)])
        # Unmatched unconfirmed tracks are deleted immediately (A.6 step 5).
        self.assertEqual(len(tracker.tracks), 0)

    def test_strong_detections_still_take_priority(self) -> None:
        tracker = self._confirmed_tracker(use_byte=True)
        track = tracker.tracks[0]
        strong = self.BOX + np.array([5.0, 0.0, 0.0, 0.0])
        tracker.predict()
        tracker.update([Detection(strong, 0.9)], [Detection(self.BOX.copy(), 0.2)])

        # Matched in stage 1/2; the weak box must not also have been applied.
        self.assertEqual(track.time_since_update, 0)
        self.assertEqual(len(tracker.tracks), 1)

    def test_byte_off_is_unaffected_by_the_new_argument(self) -> None:
        a = self._confirmed_tracker(use_byte=False)
        b = self._confirmed_tracker(use_byte=False)
        a.predict(); a.update([Detection(self.BOX.copy(), 0.9)])
        b.predict(); b.update([Detection(self.BOX.copy(), 0.9)], None)
        np.testing.assert_allclose(a.tracks[0].bbox_xywh, b.tracks[0].bbox_xywh, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
