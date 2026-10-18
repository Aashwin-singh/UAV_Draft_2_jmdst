"""Modelled ground station: precomputed detections + each frame's unloaded compute time.

Why
---
Simulating both nodes on one laptop couples them through its shared CPU/GPU
power budget: a busy CPU-only edge down-clocks the GPU the ground's YOLO runs
on (1980 -> 465 MHz measured, YOLO 15 -> 42 ms per frame, identical requests).
A real ground station is a separate machine, so live YOLO in the loop would
make every edge-hardware comparison partly a measurement of this laptop's
power management (ARCHITECTURE.md invariant 15).

What
----
The ground's output depends only on the bytes it receives: YOLO is
deterministic in the decoded image, and the edge's encoding is deterministic
in the frame. So ``scripts/build_detection_cache.py`` runs the ground's exact
path -- decode the edge's encoded frame, run YOLO with the edge's
post-processing -- once per frame **on an idle machine**, and records the
detections plus that frame's own measured decode + detect time. Entries are
keyed by the SHA-256 of the encoded image bytes, the same bytes a request
carries.

In a sweep the ground looks each request up and replies once the frame's
recorded compute time has passed since it arrived. Detections are identical
to live YOLO (verified byte-for-byte on full runs); ground compute becomes a
controlled, per-frame-realistic quantity, like the link; and the GPU is left
to the edge alone, as on separate hardware. A request that misses the cache
falls back to live YOLO and is counted, so a sweep can prove it had none.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CachedDetection:
    detections: list[tuple[np.ndarray, float]]
    compute_ms: float


class DetectionCache:
    """Detections keyed by encoded-frame hash, for one detector configuration."""

    def __init__(self, meta: dict) -> None:
        # meta must carry the post-processing the entries were computed with;
        # a request asking for anything else is a miss, never a wrong answer.
        for key in ("conf", "iou", "imgsz"):
            if key not in meta:
                raise ValueError(f"cache metadata needs {key!r}")
        self.meta = dict(meta)
        self._entries: dict[str, tuple[np.ndarray, float]] = {}

    @staticmethod
    def key(image_data: bytes) -> str:
        return hashlib.sha256(image_data).hexdigest()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, image_data: bytes, detections: list[tuple[np.ndarray, float]], compute_ms: float) -> None:
        rows = np.array([[*box, conf] for box, conf in detections], dtype=np.float64).reshape(-1, 5)
        self._entries[self.key(image_data)] = (rows, float(compute_ms))

    def lookup(self, image_data: bytes, conf: float, iou: float, imgsz: int) -> CachedDetection | None:
        if (conf, iou, imgsz) != (self.meta["conf"], self.meta["iou"], self.meta["imgsz"]):
            return None
        hit = self._entries.get(self.key(image_data))
        if hit is None:
            return None
        rows, compute_ms = hit
        return CachedDetection([(rows[i, :4].copy(), float(rows[i, 4])) for i in range(len(rows))], compute_ms)

    def compute_ms_values(self) -> np.ndarray:
        return np.array([ms for _rows, ms in self._entries.values()])

    def save(self, path: str | Path) -> None:
        keys = list(self._entries)
        counts = np.array([len(self._entries[k][0]) for k in keys], dtype=np.int64)
        rows = (np.concatenate([self._entries[k][0] for k in keys]) if keys else np.zeros((0, 5)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, keys=np.array(keys, dtype="U64"), counts=counts, rows=rows,
            compute_ms=np.array([self._entries[k][1] for k in keys]),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DetectionCache":
        data = np.load(path, allow_pickle=False)
        cache = cls(json.loads(str(data["meta"])))
        offsets = np.concatenate([[0], np.cumsum(data["counts"])])
        rows = data["rows"]
        for i, (key, ms) in enumerate(zip(data["keys"], data["compute_ms"])):
            cache._entries[str(key)] = (rows[offsets[i]:offsets[i + 1]].copy(), float(ms))
        return cache
