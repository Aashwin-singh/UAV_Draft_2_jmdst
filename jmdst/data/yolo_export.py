"""Export the unified JMDST dataset format into a YOLO-format dataset.

YOLOv11 (via ultralytics) expects, for a dataset root:
    images/<split>/*.jpg
    labels/<split>/*.txt   (same stem as the matching image)
where each label line is ``class_id x_center y_center width height``,
normalized to [0, 1] by image width/height.

Source images are not duplicated: they are hardlinked into place (falling
back to copying only if hardlinking fails, e.g. across filesystem volumes),
so exporting does not multiply disk usage.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from .io import iter_sequence_dirs, read_sequence, resolve_image_path
from .schema import CLASS_NAMES, FrameRecord, SequenceInfo


@dataclass
class YoloExportStats:
    images_written: int = 0
    labels_written: int = 0
    objects_written: int = 0
    empty_labels: int = 0
    skipped_missing_image: list[str] = field(default_factory=list)
    per_split_images: dict[str, int] = field(default_factory=dict)


def _place_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _yolo_label_lines(record: FrameRecord, image_width: int, image_height: int) -> list[str]:
    lines = []
    for obj in record.objects:
        left, top, width, height = obj.bbox_xywh
        x_center = (left + width / 2.0) / image_width
        y_center = (top + height / 2.0) / image_height
        norm_width = width / image_width
        norm_height = height / image_height
        lines.append(
            f"{obj.class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        )
    return lines


def export_yolo_split(
    unified_root: str | Path,
    output_root: str | Path,
    dataset: str,
    split: str,
    stats: YoloExportStats,
) -> None:
    sequence_dirs = iter_sequence_dirs(unified_root, dataset=dataset, split=split)
    images_dir = Path(output_root) / "images" / split
    labels_dir = Path(output_root) / "labels" / split

    count = 0
    for sequence_dir in sequence_dirs:
        info: SequenceInfo
        info, records = read_sequence(sequence_dir)
        if info.image_width is None or info.image_height is None:
            raise ValueError(f"{sequence_dir}: seqinfo.json is missing image_width/image_height")

        for record in records:
            src_image = resolve_image_path(sequence_dir, record.image_path)
            if not src_image.is_file():
                stats.skipped_missing_image.append(str(src_image))
                continue

            stem = f"{dataset}_{info.sequence}_{record.frame_id:06d}"
            dst_image = images_dir / f"{stem}{src_image.suffix}"
            _place_image(src_image, dst_image)
            stats.images_written += 1
            count += 1

            lines = _yolo_label_lines(record, info.image_width, info.image_height)
            if not lines:
                stats.empty_labels += 1
            stats.objects_written += len(lines)

            labels_dir.mkdir(parents=True, exist_ok=True)
            label_path = labels_dir / f"{stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            stats.labels_written += 1

    stats.per_split_images[split] = stats.per_split_images.get(split, 0) + count


def write_dataset_yaml(
    output_root: str | Path,
    splits: Iterable[str],
    class_names: Iterable[str] = CLASS_NAMES,
) -> Path:
    output_root = Path(output_root).resolve()
    names = {idx: name for idx, name in enumerate(class_names)}
    config = {"path": output_root.as_posix()}
    for split in splits:
        key = "val" if split == "val" else ("test" if split == "test" else "train")
        config[key] = f"images/{split}"
    config["names"] = names

    yaml_path = output_root / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return yaml_path


def export_yolo_dataset(
    unified_root: str | Path,
    output_root: str | Path,
    datasets: Iterable[str] = ("visdrone", "uavdt"),
    splits: Iterable[str] = ("train", "val", "test"),
) -> YoloExportStats:
    stats = YoloExportStats()
    present_splits: list[str] = []
    for split in splits:
        split_had_data = False
        for dataset in datasets:
            sequence_dirs = iter_sequence_dirs(unified_root, dataset=dataset, split=split)
            if not sequence_dirs:
                continue
            split_had_data = True
            export_yolo_split(unified_root, output_root, dataset, split, stats)
        if split_had_data:
            present_splits.append(split)

    write_dataset_yaml(output_root, present_splits)
    return stats
