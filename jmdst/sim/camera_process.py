"""Camera process: decode a sequence (images or a video) and publish it on a clock.

Launched by ``CameraFeed`` (jmdst/sim/camera.py) as ``python -m
jmdst.sim.camera_process``, outside the edge's Job Object. Kept separate from
camera.py so the package import does not pre-load the module being run.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from multiprocessing import shared_memory
from typing import Iterator

from .camera import _DONE, _HEADER, _LATE_US, _LATEST, _READY, _views


def _image_frames(paths: list[str], frame_bytes: int, size) -> Iterator[bytes]:
    """Decode a sequence's images ahead; PIL releases the GIL while decoding."""

    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    def load(path: str) -> bytes:
        data = Image.open(path).convert("RGB").tobytes()
        if len(data) != frame_bytes:
            raise ValueError(f"{path}: frame size differs from the sequence's {size[0]}x{size[1]}")
        return data

    window = 6
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = deque(pool.submit(load, p) for p in paths[:window])
        for k in range(len(paths)):
            data = pending.popleft().result()
            if k + window < len(paths):
                pending.append(pool.submit(load, paths[k + window]))
            yield data


def _video_frames(path: str, frame_bytes: int, size) -> Iterator[bytes]:
    """Decode a video sequentially in a reader thread, a few frames ahead."""

    import queue
    import threading

    import cv2

    frames: queue.Queue = queue.Queue(maxsize=8)

    def reader() -> None:
        capture = cv2.VideoCapture(path)
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()
                if len(data) != frame_bytes:
                    raise ValueError(f"{path}: frame size differs from {size[0]}x{size[1]}")
                frames.put(data)
        finally:
            capture.release()
            frames.put(None)

    threading.Thread(target=reader, daemon=True).start()
    while (data := frames.get()) is not None:
        yield data


def main() -> None:
    from jmdst.net.relay import high_resolution_timer

    cfg = json.loads(sys.stdin.read())
    slots = cfg["slots"]
    width, height = cfg["size"]
    frame_bytes = width * height * 3
    shm_frames = shared_memory.SharedMemory(name=cfg["frames"])
    ctl = shared_memory.SharedMemory(name=cfg["control"])
    ints, floats = _views(ctl.buf, slots)
    source = (_video_frames(cfg["video"], frame_bytes, (width, height)) if "video" in cfg
              else _image_frames(cfg["paths"], frame_bytes, (width, height)))

    with high_resolution_timer():
        first = next(source, None)  # decoded before the clock starts
        ints[_READY] = 1
        while floats[0] == 0.0:
            time.sleep(0.001)
        t0, fps = float(floats[0]), float(floats[1])
        worst = 0.0
        k, data = 0, first
        while data is not None:
            target = t0 + k / fps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            slot = k % slots
            ints[_HEADER + slots + slot] += 1  # odd: writing
            shm_frames.buf[slot * frame_bytes:(slot + 1) * frame_bytes] = data
            ints[_HEADER + slot] = k
            ints[_HEADER + slots + slot] += 1  # even: stable
            ints[_LATEST] = k
            worst = max(worst, time.perf_counter() - target)
            ints[_LATE_US] = int(worst * 1e6)
            k, data = k + 1, next(source, None)
        ints[_DONE] = 1
    del ints, floats
    shm_frames.close()
    ctl.close()


if __name__ == "__main__":
    main()
