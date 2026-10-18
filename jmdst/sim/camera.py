"""Real-time camera for the edge node: frames appear on a fixed clock.

Offline evaluation hides the cost of slow hardware: a slow edge still sees
every frame, just later, so only FPS changes and never accuracy. On a real
UAV the camera does not wait. ``CameraFeed`` publishes frame k at
``t0 + k / fps`` whether or not the edge is ready; the edge always takes the
**newest** frame, and every frame it was too busy for is skipped.

The camera runs in its **own process** (``jmdst/sim/camera_process.py``),
outside the edge's Job Object (children break away),
because it stands in for separate camera hardware: JPEG decoding is charged
to the camera, not to the edge's CPU budget. Frames are decoded ahead by a
small thread pool and published into a shared-memory ring buffer at their
capture time. Each slot carries a version counter (a seqlock), so the edge
never reads a frame half-overwritten; it retries with the newer frame instead.

Every timestamp is ``time.perf_counter`` -- one clock across processes on
this host -- so capture times compare directly with the edge's finish times.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

_LATEST, _DONE, _READY, _LATE_US, _N = 0, 1, 2, 3, 4  # control block (int64) layout
_HEADER = 8
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _control_size(slots: int) -> int:
    return (_HEADER + 2 * slots) * 8 + 2 * 8  # int64 block + float64 [t0, fps]


def _views(buf, slots: int):
    ints = np.ndarray((_HEADER + 2 * slots,), dtype=np.int64, buffer=buf)
    floats = np.ndarray((2,), dtype=np.float64, buffer=buf, offset=(_HEADER + 2 * slots) * 8)
    return ints, floats


def latest_finished(finish_times: list[float], capture_times: list[float]) -> list[int]:
    """Streaming-perception scoring (Li et al., ECCV 2020).

    For each camera frame, the index of the newest result whose finish time is
    at or before that frame's capture time, or -1 if none exists yet. Results
    are given in the order they finished, so ``finish_times`` is non-decreasing.
    A result is therefore never shown for the frame it was computed from:
    processing takes time, and the display can only show what is done.
    """

    shown, out = -1, []
    for capture in capture_times:
        while shown + 1 < len(finish_times) and finish_times[shown + 1] <= capture:
            shown += 1
        out.append(shown)
    return out


class CameraFeed:
    """Edge-side handle on a camera process replaying one sequence."""

    def __init__(self, image_paths: list[Path], size: tuple[int, int], fps: float, slots: int = 4) -> None:
        self.size = size
        self.fps = fps
        self.slots = slots
        self.n = len(image_paths)
        self.frame_bytes = size[0] * size[1] * 3
        self._frames = shared_memory.SharedMemory(create=True, size=slots * self.frame_bytes)
        self._ctl = shared_memory.SharedMemory(create=True, size=_control_size(slots))
        self._ints, self._floats = _views(self._ctl.buf, slots)
        self._ints[:] = 0
        self._ints[_LATEST] = -1
        self._ints[_N] = self.n
        self._ints[_HEADER:_HEADER + slots] = -1
        self._floats[:] = (0.0, fps)
        self.t0: float | None = None

        self._proc = subprocess.Popen(
            [sys.executable, "-m", "jmdst.sim.camera_process"], cwd=PROJECT_ROOT, stdin=subprocess.PIPE,
        )
        self._proc.stdin.write(json.dumps({
            "frames": self._frames.name, "control": self._ctl.name, "slots": slots,
            "size": list(size), "paths": [str(p) for p in image_paths],
        }).encode("utf-8"))
        self._proc.stdin.close()
        deadline = time.perf_counter() + 60
        while not self._ints[_READY]:
            if self._proc.poll() is not None:
                raise RuntimeError(f"camera process exited early (code {self._proc.returncode})")
            if time.perf_counter() > deadline:
                raise TimeoutError("camera process not ready after 60 s")
            time.sleep(0.005)

    @property
    def pid(self) -> int:
        return self._proc.pid

    def start(self, lead_s: float = 0.1) -> float:
        """Start the clock: frame 0 is captured ``lead_s`` from now."""

        self.t0 = time.perf_counter() + lead_s
        self._floats[0] = self.t0
        return self.t0

    def capture_time(self, k: int) -> float:
        return self.t0 + k / self.fps

    def take_newest(self, after: int):
        """Block until a frame newer than ``after`` exists; return (k, image).

        Returns ``None`` once the sequence has ended and nothing newer is coming.
        """

        from PIL import Image

        ints = self._ints
        while True:
            latest = int(ints[_LATEST])
            if latest > after:
                slot = latest % self.slots
                version = int(ints[_HEADER + self.slots + slot])
                if version % 2 == 0 and int(ints[_HEADER + slot]) == latest:
                    offset = slot * self.frame_bytes
                    image = Image.frombytes("RGB", self.size, bytes(self._frames.buf[offset:offset + self.frame_bytes]))
                    if int(ints[_HEADER + self.slots + slot]) == version:
                        return latest, image
                continue  # slot was being rewritten; the newer frame is now latest
            if ints[_DONE]:
                return None
            wait = self.capture_time(after + 1) - time.perf_counter()
            time.sleep(wait if wait > 0.0005 else 0.0005)

    @property
    def max_lateness_ms(self) -> float:
        """Worst publish delay behind the capture clock (camera health)."""

        return float(self._ints[_LATE_US]) / 1e3

    def close(self) -> None:
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        del self._ints, self._floats
        for shm in (self._frames, self._ctl):
            shm.close()
            shm.unlink()
