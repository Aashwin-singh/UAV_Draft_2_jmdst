"""Kalman filter for bounding-box motion prediction (paper Sec. 2.4 / A.6).

The paper's trajectory association is "improved on the basis of DeepSORT"
and uses a Kalman filter for box prediction throughout (Algorithm 1 steps 8
and 13) without specifying its internals. This is the standard DeepSORT
Kalman filter formulation (Wojke et al.): an 8-D constant-velocity state
over box center, aspect ratio, and height, observing the 4-D box directly.
We reproduce that standard baseline faithfully, since the paper gives no
reason to deviate from it.

State: [cx, cy, a, h, vx, vy, va, vh], where (cx, cy) is the box center,
a = w/h is the aspect ratio, h is the height, and the v-prefixed terms are
the respective velocities. Measurement: [cx, cy, a, h].
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def xywh_to_xyah(bbox_xywh: np.ndarray) -> np.ndarray:
    """Convert [left, top, width, height] to the Kalman measurement [cx, cy, a, h]."""

    x, y, w, h = bbox_xywh
    return np.array([x + w / 2.0, y + h / 2.0, w / h, h], dtype=np.float64)


def xyah_to_xywh(xyah: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, a, h] back to [left, top, width, height]."""

    cx, cy, a, h = xyah
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float64)


class KalmanFilter:
    """Constant-velocity Kalman filter over [cx, cy, a, h] boxes.

    Process/measurement noise scale with the tracked object's height,
    following the standard DeepSORT convention (larger objects tolerate
    proportionally larger pixel motion between frames).
    """

    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160) -> None:
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create an initial (mean, covariance) from a single detection.

        Velocity starts at zero; initial covariance is wide, especially in
        velocity, since a first observation carries no motion information.
        """

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance (mean, covariance) one time step under the motion model."""

        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project the state distribution into measurement space."""

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct (mean, covariance) with a new measurement (Kalman gain update)."""

        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), (covariance @ self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        """Squared Mahalanobis distance between the state and each measurement.

        Not used by Phase 7's track lifecycle; provided as standard Kalman
        filter tooling for Phase 8's association/gating step.
        """

        projected_mean, projected_cov = self.project(mean, covariance)
        cholesky_factor = np.linalg.cholesky(projected_cov)
        diff = measurements - projected_mean
        z = scipy.linalg.solve_triangular(cholesky_factor, diff.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)
