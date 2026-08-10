"""Camera motion compensation for the tracking loop (ECC image alignment).

Motivation
----------
Kalman state lives in the *previous* frame's pixel coordinates; detections
arrive in the *current* frame's. On a moving UAV those two frames differ by a
global transform, so a stationary car appears to jump. The paper documents
exactly this failure mode (Sec. 3.5.4, "sudden UAV maneuver") and leaves it as
future work; our own held-out test then showed UAVDT's remaining gap is
association-dominated (AssA -17.5 vs the paper, DetA -12.6), which is where
this lands.

Method is StrongSORT's ECC compensation (Du et al., IEEE TMM 2023 -- paper
ref [39], which also supplies the NSA filter in ``kalman_filter.py``):
estimate a global warp between consecutive frames with
``cv2.findTransformECC``, then push every track's Kalman state through it
before association. Image-based only -- no IMU, telemetry, or extra sensor.

StrongSORT measures +0.2 IDF1 / +0.3 MOTA from this on MOT17, whose camera is
near-static. UAV footage moves far more, so treat that as a floor rather than
an expectation.

Cost note: ECC is iterative and runs per frame. It is applied to a downscaled
grayscale image (``scale``) with a capped iteration count, since we only need
a global transform, not pixel-accurate registration.
"""

from __future__ import annotations

import numpy as np


def _to_gray(image) -> np.ndarray:
    """Accept a PIL Image or an HxW(xC) array, return a uint8 grayscale array."""

    arr = np.asarray(image.convert("L")) if hasattr(image, "convert") else np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr.astype(np.uint8, copy=False)


class CameraMotionCompensator:
    """Estimate the frame-to-frame global warp and apply it to Kalman states.

    Usage per frame, *before* ``tracker.predict()``:

        warp = cmc.estimate(image)
        if warp is not None:
            cmc.apply(tracks, warp)

    ``estimate`` returns ``None`` on the first frame and whenever ECC fails to
    converge, which is a normal outcome on low-texture or motion-blurred
    frames -- the caller simply skips compensation for that frame.
    """

    def __init__(
        self,
        scale: float = 0.25,
        max_iterations: int = 50,
        termination_eps: float = 1e-4,
        motion_type: str = "euclidean",
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.scale = float(scale)
        self.motion_type = motion_type
        modes = {
            # Rotation + translation. The default: UAV motion is dominated by
            # these two, and constraining the model keeps ECC stable.
            "euclidean": cv2.MOTION_EUCLIDEAN,
            # Adds scale/shear -- captures altitude change, but has more free
            # parameters and so is more prone to bad fits on weak texture.
            "affine": cv2.MOTION_AFFINE,
        }
        if motion_type not in modes:
            raise ValueError(f"motion_type must be one of {sorted(modes)}, got {motion_type!r}")
        self._mode = modes[motion_type]
        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(max_iterations),
            float(termination_eps),
        )
        self._prev_gray: np.ndarray | None = None
        self.failures = 0
        self.estimates = 0

    def reset(self) -> None:
        """Forget the previous frame. Call between sequences."""

        self._prev_gray = None

    def estimate(self, image) -> np.ndarray | None:
        """Return the 2x3 warp mapping previous-frame coords to this frame."""

        cv2 = self._cv2
        gray = _to_gray(image)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        prev, self._prev_gray = self._prev_gray, gray
        if prev is None or prev.shape != gray.shape:
            return None

        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _cc, warp = cv2.findTransformECC(prev, gray, warp, self._mode, self._criteria, None, 1)
        except cv2.error:
            # Non-convergence raises rather than returning a status code.
            self.failures += 1
            return None

        self.estimates += 1
        warp = warp.astype(np.float64)
        # Translation was estimated on the downscaled image; rescale it back to
        # full resolution. The linear part is scale-invariant.
        if self.scale != 1.0:
            warp[:, 2] /= self.scale
        return warp

    def apply(self, tracks, warp: np.ndarray) -> None:
        """Push every track's Kalman mean/covariance through the warp, in place.

        State layout is ``[cx, cy, a, h, vx, vy, va, vh]``. The warp's linear
        part ``R`` acts on positions and velocities; the translation acts on
        position only. Aspect ratio ``a`` is invariant under a similarity
        transform; height ``h`` is scaled by ``sqrt(det R)``, which is 1 for a
        Euclidean warp and meaningful only for an affine one.
        """

        linear = warp[:2, :2]
        translation = warp[:2, 2]
        scale = float(np.sqrt(abs(np.linalg.det(linear)))) or 1.0

        # Block-diagonal operator over the 8-D state.
        transform = np.eye(8, dtype=np.float64)
        transform[0:2, 0:2] = linear      # position
        transform[3, 3] = scale           # height
        transform[4:6, 4:6] = linear      # velocity
        transform[7, 7] = scale           # height velocity

        for track in tracks:
            mean = transform @ track.mean
            mean[0:2] += translation
            track.mean = mean
            track.covariance = transform @ track.covariance @ transform.T
