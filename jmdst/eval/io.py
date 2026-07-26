"""Load ground-truth and prediction trajectories for evaluation.

Both are represented as ``{frame_id: [(track_id, bbox_xywh), ...]}`` per
sequence. Ground truth comes from the unified format; predictions come from
the MOT-Challenge-format files written by scripts/run_jmdst.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jmdst.data.io import iter_sequence_dirs, read_sequence

FrameDict = dict[int, list[tuple[int, np.ndarray]]]


def load_ground_truth(
    unified_root: str | Path,
    dataset: str,
    split: str,
) -> dict[str, FrameDict]:
    """Load GT trajectories from the unified format, keyed by sequence name.

    The unified GT already contains only the evaluated vehicle classes (the
    converters drop everything else, including VisDrone ignore regions), and
    evaluation is class-agnostic (MOT metrics match boxes regardless of class),
    so no further class filtering is applied here.
    """

    sequences: dict[str, FrameDict] = {}
    for sequence_dir in iter_sequence_dirs(unified_root, dataset=dataset, split=split):
        info, records = read_sequence(sequence_dir)
        frames: FrameDict = {}
        for record in records:
            frames[record.frame_id] = [
                (int(obj.track_id), np.asarray(obj.bbox_xywh, dtype=np.float64))
                for obj in record.objects
            ]
        sequences[info.sequence] = frames
    return sequences


def load_mot_predictions(results_dir: str | Path) -> dict[str, FrameDict]:
    """Load MOT-format prediction files (``<sequence>.txt``) from a directory."""

    results_dir = Path(results_dir)
    sequences: dict[str, FrameDict] = {}
    for path in sorted(results_dir.glob("*.txt")):
        frames: FrameDict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            frames.setdefault(frame_id, []).append((track_id, np.array([x, y, w, h], dtype=np.float64)))
        sequences[path.stem] = frames
    return sequences
