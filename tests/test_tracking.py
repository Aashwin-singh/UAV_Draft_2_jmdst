from __future__ import annotations

import unittest

import numpy as np

from jmdst.tracking import KalmanFilter, Track, TrackState, xyah_to_xywh, xywh_to_xyah


class BBoxConversionTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        for bbox in ([10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 5.0, 5.0], [100.0, 50.0, 12.5, 60.0]):
            bbox = np.array(bbox)
            xyah = xywh_to_xyah(bbox)
            restored = xyah_to_xywh(xyah)
            np.testing.assert_allclose(restored, bbox)

    def test_xyah_semantics(self) -> None:
        bbox = np.array([10.0, 20.0, 30.0, 40.0])
        cx, cy, a, h = xywh_to_xyah(bbox)
        self.assertAlmostEqual(cx, 25.0)  # 10 + 30/2
        self.assertAlmostEqual(cy, 40.0)  # 20 + 40/2
        self.assertAlmostEqual(a, 0.75)  # 30/40
        self.assertAlmostEqual(h, 40.0)


class KalmanFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kf = KalmanFilter()

    def test_initiate_zero_velocity(self) -> None:
        measurement = xywh_to_xyah(np.array([10.0, 10.0, 20.0, 40.0]))
        mean, covariance = self.kf.initiate(measurement)
        np.testing.assert_allclose(mean[:4], measurement)
        np.testing.assert_allclose(mean[4:], np.zeros(4))
        self.assertEqual(covariance.shape, (8, 8))
        # Off-diagonal terms are zero at initiation (diagonal covariance).
        np.testing.assert_allclose(covariance, np.diag(np.diagonal(covariance)))

    def test_predict_holds_position_under_zero_velocity(self) -> None:
        measurement = xywh_to_xyah(np.array([10.0, 10.0, 20.0, 40.0]))
        mean, covariance = self.kf.initiate(measurement)
        pred_mean, pred_covariance = self.kf.predict(mean, covariance)
        # No velocity yet -> position unchanged, but uncertainty grows.
        np.testing.assert_allclose(pred_mean[:4], mean[:4])
        self.assertTrue(np.all(np.diagonal(pred_covariance) >= np.diagonal(covariance) - 1e-9))

    def test_update_moves_toward_measurement_and_shrinks_covariance(self) -> None:
        measurement = xywh_to_xyah(np.array([10.0, 10.0, 20.0, 40.0]))
        mean, covariance = self.kf.initiate(measurement)
        pred_mean, pred_covariance = self.kf.predict(mean, covariance)

        new_measurement = xywh_to_xyah(np.array([12.0, 11.0, 20.0, 40.0]))
        upd_mean, upd_covariance = self.kf.update(pred_mean, pred_covariance, new_measurement)

        # Corrected mean should land strictly between the predicted position
        # and the new measurement (a real Kalman blend, not pass-through).
        for i in range(2):
            lo, hi = sorted((pred_mean[i], new_measurement[i]))
            self.assertTrue(lo <= upd_mean[i] <= hi)
        # An update should reduce uncertainty versus the pre-update prediction.
        self.assertTrue(np.trace(upd_covariance) < np.trace(pred_covariance))

    def test_tracks_constant_velocity_motion(self) -> None:
        # A box moving +2px/frame in x: after a few predict/update cycles the
        # filter's velocity estimate should converge close to the true value.
        true_boxes = [np.array([10.0 + 2.0 * t, 10.0, 20.0, 40.0]) for t in range(6)]
        mean, covariance = self.kf.initiate(xywh_to_xyah(true_boxes[0]))
        for box in true_boxes[1:]:
            mean, covariance = self.kf.predict(mean, covariance)
            mean, covariance = self.kf.update(mean, covariance, xywh_to_xyah(box))

        self.assertAlmostEqual(mean[4], 2.0, delta=0.5)  # vx converged near 2 px/frame
        pred_mean, _ = self.kf.predict(mean, covariance)
        self.assertAlmostEqual(pred_mean[0], true_boxes[-1][0] + true_boxes[-1][2] / 2 + 2.0, delta=1.0)

    def test_gating_distance_zero_for_projected_mean(self) -> None:
        measurement = xywh_to_xyah(np.array([10.0, 10.0, 20.0, 40.0]))
        mean, covariance = self.kf.initiate(measurement)
        projected_mean, _ = self.kf.project(mean, covariance)
        distance = self.kf.gating_distance(mean, covariance, projected_mean[np.newaxis, :])
        self.assertAlmostEqual(float(distance[0]), 0.0, places=6)


class TrackStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kf = KalmanFilter()

    def _new_track(self, confirm_hits: int = 4, max_age: int = 100) -> Track:
        return Track.initiate(
            self.kf, track_id=1, bbox_xywh=np.array([10.0, 10.0, 20.0, 40.0]),
            confirm_hits=confirm_hits, max_age=max_age,
        )

    def test_starts_tentative_with_one_hit(self) -> None:
        track = self._new_track()
        self.assertTrue(track.is_tentative())
        self.assertEqual(track.hits, 1)

    def test_confirms_after_n_consecutive_updates(self) -> None:
        track = self._new_track(confirm_hits=4)
        for _ in range(2):  # hits: 1 -> 2 -> 3, still tentative
            track.predict(self.kf)
            track.update(self.kf, np.array([10.0, 10.0, 20.0, 40.0]))
            self.assertTrue(track.is_tentative())
        track.predict(self.kf)
        track.update(self.kf, np.array([10.0, 10.0, 20.0, 40.0]))  # hits: 4 -> confirmed
        self.assertTrue(track.is_confirmed())

    def test_unconfirmed_track_deleted_immediately_on_miss(self) -> None:
        track = self._new_track(confirm_hits=4)
        self.assertTrue(track.is_tentative())
        track.predict(self.kf)
        track.mark_missed()
        self.assertTrue(track.is_deleted())

    def test_confirmed_track_survives_misses_up_to_max_age(self) -> None:
        track = self._new_track(confirm_hits=2, max_age=5)
        track.predict(self.kf)
        track.update(self.kf, np.array([10.0, 10.0, 20.0, 40.0]))  # hits=2 -> confirmed
        self.assertTrue(track.is_confirmed())

        for _ in range(5):  # time_since_update goes 1..5, still <= max_age
            track.predict(self.kf)
            track.mark_missed()
            self.assertTrue(track.is_confirmed())

        track.predict(self.kf)  # time_since_update = 6 > max_age(5)
        track.mark_missed()
        self.assertTrue(track.is_deleted())

    def test_update_resets_miss_counter_before_deletion(self) -> None:
        track = self._new_track(confirm_hits=2, max_age=3)
        track.predict(self.kf)
        track.update(self.kf, np.array([10.0, 10.0, 20.0, 40.0]))  # confirmed

        for _ in range(3):
            track.predict(self.kf)
            track.mark_missed()
        self.assertTrue(track.is_confirmed())  # time_since_update == max_age, not deleted yet

        # A fresh update resets the miss counter -- track survives well past
        # what would have been the deletion point without it.
        track.predict(self.kf)
        track.update(self.kf, np.array([11.0, 10.0, 20.0, 40.0]))
        self.assertEqual(track.time_since_update, 0)
        for _ in range(3):
            track.predict(self.kf)
            track.mark_missed()
            self.assertTrue(track.is_confirmed())

    def test_embedding_history_accumulates_in_order(self) -> None:
        embeddings = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        track = Track.initiate(
            self.kf, track_id=1, bbox_xywh=np.array([10.0, 10.0, 20.0, 40.0]),
            confirm_hits=10, embedding=embeddings[0],
        )
        for emb in embeddings[1:]:
            track.predict(self.kf)
            track.update(self.kf, np.array([10.0, 10.0, 20.0, 40.0]), embedding=emb)

        self.assertEqual(len(track.embedding_history), 3)
        for actual, expected in zip(track.embedding_history, embeddings):
            np.testing.assert_array_equal(actual, expected)

    def test_bbox_xywh_round_trips_through_initiate(self) -> None:
        bbox = np.array([10.0, 20.0, 30.0, 40.0])
        track = Track.initiate(self.kf, track_id=1, bbox_xywh=bbox)
        np.testing.assert_allclose(track.bbox_xywh, bbox, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
