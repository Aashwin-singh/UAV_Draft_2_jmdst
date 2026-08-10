"""Tests for ECC camera-motion compensation (jmdst/tracking/cmc.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.tracking import CameraMotionCompensator, KalmanFilter, Track


def canvas(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """A large smoothly-structured image to crop frames out of.

    ECC is gradient-based: it needs smooth structure, not per-pixel noise.
    Cropping (rather than np.roll) also avoids wrap-around seams, which
    otherwise dominate the alignment.
    """

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    img = 120 + 60 * np.sin(xx / 23.0) * np.cos(yy / 31.0) + 40 * np.sin((xx + yy) / 17.0)
    for _ in range(60):
        cx, cy, r = rng.integers(0, width), rng.integers(0, height), rng.integers(8, 26)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] += rng.integers(-50, 50)
    return np.clip(img, 0, 255).astype(np.uint8)


# Crop geometry shared by the estimate tests.
X0, Y0, CROP_W, CROP_H = 200, 150, 320, 240


def frame(source: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Crop the window at (X0+dx, Y0+dy).

    Moving the window by (dx, dy) moves the image *content* by (-dx, -dy),
    which is what the estimated warp reports -- verified against
    cv2.findTransformECC's convention.
    """

    return source[Y0 + dy: Y0 + dy + CROP_H, X0 + dx: X0 + dx + CROP_W]


class CmcEstimateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = canvas()

    def test_first_frame_returns_none(self) -> None:
        cmc = CameraMotionCompensator(scale=1.0)
        self.assertIsNone(cmc.estimate(frame(self.canvas)))

    def test_recovers_a_known_translation(self) -> None:
        cmc = CameraMotionCompensator(scale=1.0, max_iterations=500, termination_eps=1e-7)
        cmc.estimate(frame(self.canvas))
        warp = cmc.estimate(frame(self.canvas, dx=10, dy=-6))

        self.assertIsNotNone(warp)
        # Window moved (+10, -6), so content -- and any world-static target --
        # moves (-10, +6). The warp must report the content motion, since that
        # is what apply() adds to each track's position.
        np.testing.assert_allclose(warp[0, 2], -10.0, atol=1.0)
        np.testing.assert_allclose(warp[1, 2], 6.0, atol=1.0)

    def test_identical_frames_give_near_identity(self) -> None:
        cmc = CameraMotionCompensator(scale=1.0, max_iterations=200)
        cmc.estimate(frame(self.canvas))
        warp = cmc.estimate(frame(self.canvas))

        self.assertIsNotNone(warp)
        np.testing.assert_allclose(warp, np.eye(2, 3), atol=1e-2)

    def test_downscaling_rescales_translation_back_to_full_resolution(self) -> None:
        full = CameraMotionCompensator(scale=1.0, max_iterations=500, termination_eps=1e-7)
        half = CameraMotionCompensator(scale=0.5, max_iterations=500, termination_eps=1e-7)
        for c in (full, half):
            c.estimate(frame(self.canvas))
        moved = frame(self.canvas, dx=8)
        w_full, w_half = full.estimate(moved), half.estimate(moved)

        self.assertIsNotNone(w_half)
        # Both must report the shift in full-resolution pixels.
        self.assertAlmostEqual(w_full[0, 2], w_half[0, 2], delta=1.5)

    def test_reset_forgets_previous_frame(self) -> None:
        cmc = CameraMotionCompensator(scale=1.0)
        cmc.estimate(frame(self.canvas))
        cmc.reset()
        self.assertIsNone(cmc.estimate(frame(self.canvas)))

    def test_shape_change_is_treated_as_a_fresh_start(self) -> None:
        cmc = CameraMotionCompensator(scale=1.0)
        cmc.estimate(frame(self.canvas))
        self.assertIsNone(cmc.estimate(self.canvas[0:120, 0:160]))

    def test_world_static_target_stays_on_its_content(self) -> None:
        """End-to-end sign check: the whole point of CMC.

        A target fixed in the world sits at a fixed canvas position. After the
        camera moves, applying the warp must move its track box to where that
        content now is -- not the other way round.
        """

        cmc = CameraMotionCompensator(scale=1.0, max_iterations=500, termination_eps=1e-7)
        kf = KalmanFilter()
        # Target at canvas (X0+120, Y0+90) -> frame coords (120, 90) initially.
        track = Track.initiate(kf, track_id=1, bbox_xywh=np.array([110.0, 80.0, 20.0, 20.0]))

        cmc.estimate(frame(self.canvas))
        warp = cmc.estimate(frame(self.canvas, dx=10, dy=-6))
        cmc.apply([track], warp)

        # Same canvas content now sits 10px left and 6px down in the new frame.
        np.testing.assert_allclose(track.bbox_xywh[0], 100.0, atol=1.0)
        np.testing.assert_allclose(track.bbox_xywh[1], 86.0, atol=1.0)

    def test_rejects_unknown_motion_type(self) -> None:
        with self.assertRaises(ValueError):
            CameraMotionCompensator(motion_type="homography")


class CmcApplyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kf = KalmanFilter()
        self.cmc = CameraMotionCompensator(scale=1.0)

    def _track(self, bbox=(100.0, 100.0, 20.0, 40.0)) -> Track:
        return Track.initiate(self.kf, track_id=1, bbox_xywh=np.array(bbox))

    def test_pure_translation_shifts_box_position(self) -> None:
        track = self._track()
        before = track.bbox_xywh.copy()
        warp = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, -7.0]])
        self.cmc.apply([track], warp)
        after = track.bbox_xywh

        np.testing.assert_allclose(after[0], before[0] + 15.0, atol=1e-9)
        np.testing.assert_allclose(after[1], before[1] - 7.0, atol=1e-9)
        # Size must be untouched by a pure translation.
        np.testing.assert_allclose(after[2:], before[2:], atol=1e-9)

    def test_identity_warp_is_a_no_op(self) -> None:
        track = self._track()
        before = track.bbox_xywh.copy()
        self.cmc.apply([track], np.eye(2, 3))
        np.testing.assert_allclose(track.bbox_xywh, before, atol=1e-9)

    def test_affine_scale_scales_box_height(self) -> None:
        track = self._track()
        before = track.bbox_xywh.copy()
        warp = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        self.cmc.apply([track], warp)

        np.testing.assert_allclose(track.bbox_xywh[3], before[3] * 2.0, rtol=1e-9)

    def test_covariance_stays_symmetric_positive_definite(self) -> None:
        track = self._track()
        warp = np.array([[0.98, -0.17, 12.0], [0.17, 0.98, -5.0]])
        self.cmc.apply([track], warp)

        cov = track.covariance
        np.testing.assert_allclose(cov, cov.T, atol=1e-9)
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) > 0))

    def test_applies_to_every_track(self) -> None:
        tracks = [
            Track.initiate(self.kf, track_id=i, bbox_xywh=np.array([10.0 * i, 50.0, 20.0, 40.0]))
            for i in range(1, 4)
        ]
        before = [t.bbox_xywh.copy() for t in tracks]
        self.cmc.apply(tracks, np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]]))
        for t, b in zip(tracks, before):
            np.testing.assert_allclose(t.bbox_xywh[0], b[0] + 5.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
