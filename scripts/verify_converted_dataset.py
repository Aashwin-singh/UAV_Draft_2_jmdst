"""Verify a unified JMDST dataset conversion.

The verifier checks the converted JSONL annotations, image references, bbox
validity, track-id consistency, image decoding, and 64x64 SSI crop generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.data.crops import CropConfig, crop_ssi
from jmdst.data.io import ANNOTATIONS_NAME, SEQINFO_NAME, resolve_image_path
from jmdst.data.schema import FrameRecord, ObjectAnnotation, SequenceInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_converted_dataset")


@dataclass
class VerificationStats:
    sequence_dirs: int = 0
    valid_sequences: int = 0
    frames: int = 0
    annotations: int = 0
    empty_frames: int = 0
    unique_track_keys: set[str] = field(default_factory=set)
    raw_track_ids_by_sequence: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    class_distribution: Counter[str] = field(default_factory=Counter)
    bbox_widths: list[float] = field(default_factory=list)
    bbox_heights: list[float] = field(default_factory=list)
    bbox_areas: list[float] = field(default_factory=list)
    bbox_aspects: list[float] = field(default_factory=list)
    ssi_crops_generated: int = 0
    missing_images: list[dict[str, Any]] = field(default_factory=list)
    missing_annotations: list[dict[str, Any]] = field(default_factory=list)
    corrupted_files: list[dict[str, Any]] = field(default_factory=list)
    invalid_bboxes: list[dict[str, Any]] = field(default_factory=list)
    out_of_bounds_bboxes: list[dict[str, Any]] = field(default_factory=list)
    track_inconsistencies: list[dict[str, Any]] = field(default_factory=list)
    invalid_ssi_crops: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unique_track_ids(self) -> int:
        return len(self.unique_track_keys)

    @property
    def ready_for_yolov11(self) -> bool:
        return not (
            self.missing_images
            or self.missing_annotations
            or self.corrupted_files
            or self.invalid_bboxes
            or self.out_of_bounds_bboxes
            or self.parse_errors
        )


def _discover_sequence_dirs(
    unified_root: Path,
    dataset: str | None = None,
    split: str | None = None,
) -> list[Path]:
    dataset_pattern = dataset if dataset else "*"
    split_pattern = split if split else "*"
    return sorted(path for path in unified_root.glob(f"{dataset_pattern}/{split_pattern}/*") if path.is_dir())


def _load_seqinfo(sequence_dir: Path, stats: VerificationStats) -> SequenceInfo | None:
    seqinfo_path = sequence_dir / SEQINFO_NAME
    if not seqinfo_path.is_file():
        stats.corrupted_files.append(
            {"path": str(seqinfo_path), "reason": "missing seqinfo.json"}
        )
        return None
    try:
        with seqinfo_path.open("r", encoding="utf-8") as handle:
            return SequenceInfo.from_dict(json.load(handle))
    except Exception as exc:
        stats.corrupted_files.append(
            {"path": str(seqinfo_path), "reason": f"failed to parse seqinfo: {exc}"}
        )
        return None


def _load_records(
    sequence_dir: Path,
    sequence_key: str,
    stats: VerificationStats,
) -> list[FrameRecord] | None:
    annotation_path = sequence_dir / ANNOTATIONS_NAME
    if not annotation_path.is_file():
        stats.missing_annotations.append(
            {"sequence": sequence_key, "path": str(annotation_path), "reason": "missing annotations.jsonl"}
        )
        return None

    records: list[FrameRecord] = []
    try:
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(FrameRecord.from_dict(json.loads(line)))
                except Exception as exc:
                    stats.parse_errors.append(
                        {
                            "sequence": sequence_key,
                            "path": str(annotation_path),
                            "line": line_number,
                            "reason": str(exc),
                        }
                    )
    except Exception as exc:
        stats.corrupted_files.append(
            {"path": str(annotation_path), "reason": f"failed to read annotations: {exc}"}
        )
        return None
    return records


def _load_image(image_path: Path, sequence_key: str, frame_id: int, stats: VerificationStats) -> Image.Image | None:
    if not image_path.is_file():
        stats.missing_images.append(
            {
                "sequence": sequence_key,
                "frame_id": frame_id,
                "path": str(image_path),
            }
        )
        return None

    try:
        image = Image.open(image_path)
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        stats.corrupted_files.append(
            {
                "sequence": sequence_key,
                "frame_id": frame_id,
                "path": str(image_path),
                "reason": f"failed to decode image: {exc}",
            }
        )
        return None


def _is_finite_bbox(obj: ObjectAnnotation) -> bool:
    return all(math.isfinite(float(value)) for value in obj.bbox_xywh)


def _validate_bbox(
    obj: ObjectAnnotation,
    image_size: tuple[int, int] | None,
    sequence_key: str,
    frame_id: int,
    stats: VerificationStats,
) -> bool:
    if not _is_finite_bbox(obj):
        stats.invalid_bboxes.append(
            {
                "sequence": sequence_key,
                "frame_id": frame_id,
                "track_id": obj.track_id,
                "bbox_xywh": obj.bbox_xywh,
                "reason": "bbox contains non-finite values",
            }
        )
        return False

    x, y, width, height = [float(value) for value in obj.bbox_xywh]
    if width <= 0 or height <= 0:
        stats.invalid_bboxes.append(
            {
                "sequence": sequence_key,
                "frame_id": frame_id,
                "track_id": obj.track_id,
                "bbox_xywh": obj.bbox_xywh,
                "reason": "bbox width/height must be positive",
            }
        )
        return False

    if image_size is not None:
        image_width, image_height = image_size
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            stats.out_of_bounds_bboxes.append(
                {
                    "sequence": sequence_key,
                    "frame_id": frame_id,
                    "track_id": obj.track_id,
                    "bbox_xywh": obj.bbox_xywh,
                    "image_size": [image_width, image_height],
                    "reason": "bbox extends outside image boundaries",
                }
            )

    stats.bbox_widths.append(width)
    stats.bbox_heights.append(height)
    stats.bbox_areas.append(width * height)
    stats.bbox_aspects.append(width / height)
    return True


def _check_track_consistency(
    sequence_key: str,
    frame: FrameRecord,
    track_classes: dict[int, tuple[int, str]],
    stats: VerificationStats,
) -> None:
    ids_in_frame: set[int] = set()
    for obj in frame.objects:
        if obj.track_id in ids_in_frame:
            stats.track_inconsistencies.append(
                {
                    "sequence": sequence_key,
                    "frame_id": frame.frame_id,
                    "track_id": obj.track_id,
                    "reason": "duplicate track id in one frame",
                }
            )
        ids_in_frame.add(obj.track_id)

        current_class = (obj.class_id, obj.class_name)
        previous_class = track_classes.get(obj.track_id)
        if previous_class is None:
            track_classes[obj.track_id] = current_class
        elif previous_class != current_class:
            stats.track_inconsistencies.append(
                {
                    "sequence": sequence_key,
                    "frame_id": frame.frame_id,
                    "track_id": obj.track_id,
                    "previous_class": previous_class,
                    "current_class": current_class,
                    "reason": "track id changes class within sequence",
                }
            )
            track_classes[obj.track_id] = current_class


def _verify_ssi(
    image: Image.Image,
    obj: ObjectAnnotation,
    sequence_key: str,
    frame_id: int,
    stats: VerificationStats,
    crop_config: CropConfig,
) -> None:
    try:
        ssi, _ = crop_ssi(image, obj.bbox_xywh, config=crop_config)
        stats.ssi_crops_generated += 1
        if ssi.size != (64, 64):
            stats.invalid_ssi_crops.append(
                {
                    "sequence": sequence_key,
                    "frame_id": frame_id,
                    "track_id": obj.track_id,
                    "size": list(ssi.size),
                }
            )
    except Exception as exc:
        stats.invalid_ssi_crops.append(
            {
                "sequence": sequence_key,
                "frame_id": frame_id,
                "track_id": obj.track_id,
                "bbox_xywh": obj.bbox_xywh,
                "reason": str(exc),
            }
        )


def verify_unified_dataset(
    unified_root: str | Path,
    dataset: str | None = None,
    split: str | None = None,
    verify_ssi: bool = True,
    progress_every: int = 2000,
    log_progress: bool = False,
) -> VerificationStats:
    """Walk the unified dataset and validate it.

    When ``log_progress`` is True, this logs a running status line every
    ``progress_every`` processed object annotations, and once per finished
    sequence, so a long full-SSI verification run stays visible instead of
    printing only a single report at the very end.
    """

    unified_root = Path(unified_root)
    stats = VerificationStats()
    crop_config = CropConfig(output_size=64, random_shift_px=0.0)

    sequence_dirs = _discover_sequence_dirs(unified_root, dataset=dataset, split=split)
    stats.sequence_dirs = len(sequence_dirs)

    start_time = time.monotonic()
    last_logged_annotations = 0

    if log_progress:
        logger.info(
            "Starting verification: %d sequence directories under %s (verify_ssi=%s)",
            stats.sequence_dirs,
            unified_root,
            verify_ssi,
        )

    for seq_index, sequence_dir in enumerate(sequence_dirs, start=1):
        info = _load_seqinfo(sequence_dir, stats)
        fallback_sequence = sequence_dir.name
        fallback_split = sequence_dir.parent.name
        fallback_dataset = sequence_dir.parent.parent.name
        sequence_key = f"{fallback_dataset}/{fallback_split}/{fallback_sequence}"
        if info is not None:
            sequence_key = f"{info.dataset}/{info.split}/{info.sequence}"

        records = _load_records(sequence_dir, sequence_key, stats)
        if records is None:
            continue

        stats.valid_sequences += 1
        stats.frames += len(records)
        if info is not None and info.frame_count != len(records):
            missing = max(0, info.frame_count - len(records))
            if missing:
                stats.missing_annotations.append(
                    {
                        "sequence": sequence_key,
                        "expected_frames": info.frame_count,
                        "actual_annotation_records": len(records),
                        "missing_frame_records": missing,
                        "reason": "seqinfo frame_count exceeds annotations.jsonl records",
                    }
                )

        frame_ids_seen: set[int] = set()
        track_classes: dict[int, tuple[int, str]] = {}

        for record in records:
            if record.frame_id in frame_ids_seen:
                stats.parse_errors.append(
                    {
                        "sequence": sequence_key,
                        "frame_id": record.frame_id,
                        "reason": "duplicate frame record",
                    }
                )
            frame_ids_seen.add(record.frame_id)

            if not record.objects:
                stats.empty_frames += 1

            image_path = resolve_image_path(sequence_dir, record.image_path)
            image = _load_image(image_path, sequence_key, record.frame_id, stats)
            image_size = image.size if image is not None else None

            _check_track_consistency(sequence_key, record, track_classes, stats)

            for obj in record.objects:
                stats.annotations += 1
                stats.class_distribution[obj.class_name] += 1
                stats.unique_track_keys.add(f"{sequence_key}:{obj.track_id}")
                stats.raw_track_ids_by_sequence[sequence_key].add(obj.track_id)

                valid_bbox = _validate_bbox(
                    obj,
                    image_size=image_size,
                    sequence_key=sequence_key,
                    frame_id=record.frame_id,
                    stats=stats,
                )
                if verify_ssi and image is not None and valid_bbox:
                    _verify_ssi(image, obj, sequence_key, record.frame_id, stats, crop_config)

                if log_progress and stats.annotations - last_logged_annotations >= progress_every:
                    elapsed = time.monotonic() - start_time
                    rate = stats.annotations / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "seq %d/%d [%s] frames=%d annotations=%d ssi_crops=%d issues=%d (%.0f obj/s)",
                        seq_index,
                        len(sequence_dirs),
                        sequence_key,
                        stats.frames,
                        stats.annotations,
                        stats.ssi_crops_generated,
                        len(stats.missing_images)
                        + len(stats.corrupted_files)
                        + len(stats.invalid_bboxes)
                        + len(stats.out_of_bounds_bboxes)
                        + len(stats.invalid_ssi_crops)
                        + len(stats.parse_errors),
                        rate,
                    )
                    last_logged_annotations = stats.annotations

        if log_progress:
            elapsed = time.monotonic() - start_time
            logger.info(
                "finished sequence %d/%d [%s]: %d frames, %d annotations so far, %.1fs elapsed",
                seq_index,
                len(sequence_dirs),
                sequence_key,
                len(records),
                stats.annotations,
                elapsed,
            )

    return stats


def _describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "min": ordered[0],
        "mean": mean(values),
        "median": median(values),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _sample(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return items[: max(0, limit)]


def stats_to_dict(stats: VerificationStats, sample_limit: int = 20) -> dict[str, Any]:
    return {
        "sequence_dirs": stats.sequence_dirs,
        "valid_sequences": stats.valid_sequences,
        "frames": stats.frames,
        "annotations": stats.annotations,
        "empty_frames": stats.empty_frames,
        "unique_track_ids_sequence_scoped": stats.unique_track_ids,
        "class_distribution": dict(stats.class_distribution),
        "bbox_stats": {
            "width": _describe(stats.bbox_widths),
            "height": _describe(stats.bbox_heights),
            "area": _describe(stats.bbox_areas),
            "aspect_ratio": _describe(stats.bbox_aspects),
        },
        "ssi_crops_generated": stats.ssi_crops_generated,
        "missing_images": len(stats.missing_images),
        "missing_annotations": len(stats.missing_annotations),
        "corrupted_files": len(stats.corrupted_files),
        "invalid_bboxes": len(stats.invalid_bboxes),
        "out_of_bounds_bboxes": len(stats.out_of_bounds_bboxes),
        "track_inconsistencies": len(stats.track_inconsistencies),
        "invalid_ssi_crops": len(stats.invalid_ssi_crops),
        "parse_errors": len(stats.parse_errors),
        "ready_for_yolov11": stats.ready_for_yolov11,
        "samples": {
            "missing_images": _sample(stats.missing_images, sample_limit),
            "missing_annotations": _sample(stats.missing_annotations, sample_limit),
            "corrupted_files": _sample(stats.corrupted_files, sample_limit),
            "invalid_bboxes": _sample(stats.invalid_bboxes, sample_limit),
            "out_of_bounds_bboxes": _sample(stats.out_of_bounds_bboxes, sample_limit),
            "track_inconsistencies": _sample(stats.track_inconsistencies, sample_limit),
            "invalid_ssi_crops": _sample(stats.invalid_ssi_crops, sample_limit),
            "parse_errors": _sample(stats.parse_errors, sample_limit),
        },
    }


def format_report(
    stats: VerificationStats,
    unified_root: str | Path,
    dataset: str | None,
    split: str | None,
    sample_limit: int = 10,
) -> str:
    bbox = stats_to_dict(stats, sample_limit=sample_limit)["bbox_stats"]
    class_rows = "\n".join(
        f"- {name}: {count}" for name, count in sorted(stats.class_distribution.items())
    ) or "- none"

    issue_rows = [
        ("Missing images", len(stats.missing_images)),
        ("Missing annotations", len(stats.missing_annotations)),
        ("Corrupted files", len(stats.corrupted_files)),
        ("Invalid bounding boxes", len(stats.invalid_bboxes)),
        ("Out-of-bounds bounding boxes", len(stats.out_of_bounds_bboxes)),
        ("Track inconsistencies", len(stats.track_inconsistencies)),
        ("Invalid SSI crops", len(stats.invalid_ssi_crops)),
        ("Parse errors", len(stats.parse_errors)),
    ]
    issue_text = "\n".join(f"- {name}: {count}" for name, count in issue_rows)

    bbox_text = "\n".join(
        (
            "| metric | min | mean | median | p95 | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| width | {_fmt(bbox['width']['min'])} | {_fmt(bbox['width']['mean'])} | {_fmt(bbox['width']['median'])} | {_fmt(bbox['width']['p95'])} | {_fmt(bbox['width']['max'])} |",
            f"| height | {_fmt(bbox['height']['min'])} | {_fmt(bbox['height']['mean'])} | {_fmt(bbox['height']['median'])} | {_fmt(bbox['height']['p95'])} | {_fmt(bbox['height']['max'])} |",
            f"| area | {_fmt(bbox['area']['min'])} | {_fmt(bbox['area']['mean'])} | {_fmt(bbox['area']['median'])} | {_fmt(bbox['area']['p95'])} | {_fmt(bbox['area']['max'])} |",
            f"| aspect ratio | {_fmt(bbox['aspect_ratio']['min'])} | {_fmt(bbox['aspect_ratio']['mean'])} | {_fmt(bbox['aspect_ratio']['median'])} | {_fmt(bbox['aspect_ratio']['p95'])} | {_fmt(bbox['aspect_ratio']['max'])} |",
        )
    )

    readiness = "READY" if stats.ready_for_yolov11 else "NOT READY"
    scope = f"dataset={dataset or '*'}, split={split or '*'}"
    return f"""# Converted Dataset Verification Report

Root: `{Path(unified_root)}`
Scope: `{scope}`

## Summary

- Sequence directories: {stats.sequence_dirs}
- Valid sequences: {stats.valid_sequences}
- Frames: {stats.frames}
- Annotations: {stats.annotations}
- Empty frames / missing frame annotations: {stats.empty_frames}
- Unique track IDs, sequence-scoped: {stats.unique_track_ids}
- SSI crops generated: {stats.ssi_crops_generated}
- YOLOv11 training readiness: **{readiness}**

## Class Distribution

{class_rows}

## Bounding Box Size Statistics

{bbox_text}

## Verification Issues

{issue_text}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. visdrone or uavdt.")
    parser.add_argument("--split", default=None, help="Optional split filter, e.g. train.")
    parser.add_argument(
        "--skip-ssi",
        action="store_true",
        help="Skip generating in-memory SSI crops. Faster, but does not verify 64x64 SSI output.",
    )
    parser.add_argument("--report-path", default=None, help="Optional markdown report path.")
    parser.add_argument("--json-path", default=None, help="Optional JSON details path.")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Log a status line every N object annotations processed.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the periodic progress log lines (report still prints at the end).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = verify_unified_dataset(
        unified_root=args.unified_root,
        dataset=args.dataset,
        split=args.split,
        verify_ssi=not args.skip_ssi,
        progress_every=args.progress_every,
        log_progress=not args.no_progress,
    )
    report = format_report(
        stats,
        unified_root=args.unified_root,
        dataset=args.dataset,
        split=args.split,
        sample_limit=args.sample_limit,
    )
    print(report)

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(stats_to_dict(stats, sample_limit=args.sample_limit), indent=2),
            encoding="utf-8",
        )

    if not stats.ready_for_yolov11:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
