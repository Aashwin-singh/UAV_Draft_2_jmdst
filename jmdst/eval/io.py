"""Load ground-truth and prediction trajectories for evaluation.

Both are represented as ``{frame_id: [(track_id, bbox_xywh), ...]}`` per
sequence. Ground truth comes from the unified format; predictions come from
the MOT-Challenge-format files written by scripts/run_jmdst.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jmdst.data.io import iter_sequence_dirs, read_sequence
from jmdst.tracking.matching import iou_xywh

FrameDict = dict[int, list[tuple[int, np.ndarray]]]
# {sequence: {frame_id: [ignore_bbox_xywh, ...]}}
IgnoreDict = dict[str, dict[int, list[np.ndarray]]]


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


def load_ignore_regions(
    unified_root: str | Path,
    dataset: str,
    split: str,
) -> IgnoreDict:
    """Load per-frame ignore regions (``ignore_regions.jsonl``) if present.

    Only UAVDT ships these; sequences without the file contribute nothing.
    """

    ignore: IgnoreDict = {}
    for sequence_dir in iter_sequence_dirs(unified_root, dataset=dataset, split=split):
        path = sequence_dir / "ignore_regions.jsonl"
        if not path.is_file():
            continue
        frames: dict[int, list[np.ndarray]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            frames[int(record["frame_id"])] = [
                np.asarray(r, dtype=np.float64) for r in record["regions_xywh"]
            ]
        ignore[sequence_dir.name] = frames
    return ignore


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


def _intersection_over_pred_area(box: np.ndarray, regions: np.ndarray) -> float:
    """Max over regions of intersection(box, region) / area(box)."""

    bx1, by1, bx2, by2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
    rx1, ry1 = regions[:, 0], regions[:, 1]
    rx2, ry2 = regions[:, 0] + regions[:, 2], regions[:, 1] + regions[:, 3]
    ix1 = np.maximum(bx1, rx1)
    iy1 = np.maximum(by1, ry1)
    ix2 = np.minimum(bx2, rx2)
    iy2 = np.minimum(by2, ry2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = max(box[2] * box[3], 1e-9)
    return float(np.max(inter / area)) if len(regions) else 0.0


def filter_predictions_by_ignore(
    predictions: dict[str, FrameDict],
    ground_truth: dict[str, FrameDict],
    ignore: IgnoreDict,
    ioa_threshold: float = 0.5,
    gt_iou_threshold: float = 0.5,
) -> tuple[dict[str, FrameDict], int]:
    """Drop predictions that fall inside ignore regions and match no real GT.

    Implements the UAVDT ignore protocol: a prediction whose area is mostly
    (> ``ioa_threshold``) inside an ignore region is discarded, but only if it
    does not also match a real GT box (IoU >= ``gt_iou_threshold``) -- so true
    positives are never removed, just would-be false positives in don't-care
    areas. Returns the filtered predictions and the number of boxes removed.
    """

    filtered: dict[str, FrameDict] = {}
    removed = 0
    for seq, frames in predictions.items():
        seq_ignore = ignore.get(seq, {})
        seq_gt = ground_truth.get(seq, {})
        new_frames: FrameDict = {}
        for frame_id, objs in frames.items():
            regions = seq_ignore.get(frame_id, [])
            if not regions:
                new_frames[frame_id] = objs
                continue
            region_arr = np.array(regions, dtype=np.float64).reshape(-1, 4)
            gt_boxes = np.array([b for _, b in seq_gt.get(frame_id, [])], dtype=np.float64).reshape(-1, 4)
            kept = []
            for track_id, box in objs:
                in_ignore = _intersection_over_pred_area(box, region_arr) > ioa_threshold
                matches_gt = len(gt_boxes) > 0 and float(np.max(iou_xywh(box, gt_boxes))) >= gt_iou_threshold
                if in_ignore and not matches_gt:
                    removed += 1
                    continue
                kept.append((track_id, box))
            new_frames[frame_id] = kept
        filtered[seq] = new_frames
    return filtered, removed
