"""Camera process: decode one sequence's frames and publish them on a clock.

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

from .camera import _DONE, _HEADER, _LATE_US, _LATEST, _READY, _views


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from jmdst.net.relay import high_resolution_timer

    cfg = json.loads(sys.stdin.read())
    slots = cfg["slots"]
    width, height = cfg["size"]
    frame_bytes = width * height * 3
    frames = shared_memory.SharedMemory(name=cfg["frames"])
    ctl = shared_memory.SharedMemory(name=cfg["control"])
    ints, floats = _views(ctl.buf, slots)
    paths = cfg["paths"]

    def load(path: str) -> bytes:
        data = Image.open(path).convert("RGB").tobytes()
        if len(data) != frame_bytes:
            raise ValueError(f"{path}: frame size differs from the sequence's {width}x{height}")
        return data

    window = 6  # frames decoded ahead; PIL releases the GIL while decoding
    with ThreadPoolExecutor(max_workers=3) as pool, high_resolution_timer():
        pending = deque(pool.submit(load, p) for p in paths[:window])
        pending[0].result()
        ints[_READY] = 1
        while floats[0] == 0.0:
            time.sleep(0.001)
        t0, fps = float(floats[0]), float(floats[1])
        worst = 0.0
        for k in range(len(paths)):
            data = pending.popleft().result()
            if k + window < len(paths):
                pending.append(pool.submit(load, paths[k + window]))
            target = t0 + k / fps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            slot = k % slots
            ints[_HEADER + slots + slot] += 1  # odd: writing
            frames.buf[slot * frame_bytes:(slot + 1) * frame_bytes] = data
            ints[_HEADER + slot] = k
            ints[_HEADER + slots + slot] += 1  # even: stable
            ints[_LATEST] = k
            worst = max(worst, time.perf_counter() - target)
            ints[_LATE_US] = int(worst * 1e6)
        ints[_DONE] = 1
    del ints, floats
    frames.close()
    ctl.close()


if __name__ == "__main__":
    main()
