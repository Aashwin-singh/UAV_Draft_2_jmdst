"""Shared converter helpers."""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

from jmdst.data.schema import FrameRecord


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def natural_frame_id(path: Path) -> int:
    """Extract a numeric frame id from an image filename."""

    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"Cannot infer frame id from image filename: {path}")
    return int(matches[-1])


def list_images(image_dir: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{ext}"):
            images[natural_frame_id(path)] = path
    return dict(sorted(images.items()))


def image_size(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    with Image.open(path) as image:
        return image.size


def clip_bbox_xywh(
    bbox_xywh: list[float],
    image_width: int | None,
    image_height: int | None,
) -> list[float] | None:
    """Clip an xywh bbox to image bounds. Returns None if nothing remains.

    Source MOT annotations occasionally extend slightly past the image edge
    for partially-visible targets at frame boundaries. Clip rather than drop
    the whole detection, since the target is still a valid training example.
    """

    if image_width is None or image_height is None:
        return bbox_xywh

    x, y, width, height = bbox_xywh
    x1 = max(0.0, min(x, float(image_width)))
    y1 = max(0.0, min(y, float(image_height)))
    x2 = max(0.0, min(x + width, float(image_width)))
    y2 = max(0.0, min(y + height, float(image_height)))

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return [x1, y1, clipped_width, clipped_height]


def group_records_by_frame(
    images_by_frame: dict[int, Path],
    annotations_by_frame: dict[int, list],
    output_sequence_dir: Path,
    copy_images: bool = False,
) -> list[FrameRecord]:
    """Build frame records and optionally copy images into the unified folder."""

    records: list[FrameRecord] = []
    copied_dir = output_sequence_dir / "images"
    if copy_images:
        copied_dir.mkdir(parents=True, exist_ok=True)

    for frame_id, src_image_path in images_by_frame.items():
        if copy_images:
            dst_path = copied_dir / src_image_path.name
            if not dst_path.exists():
                shutil.copy2(src_image_path, dst_path)
            image_path = f"images/{dst_path.name}"
        else:
            image_path = src_image_path.resolve().as_posix()

        records.append(
            FrameRecord(
                frame_id=frame_id,
                image_path=image_path,
                objects=annotations_by_frame.get(frame_id, []),
            )
        )
    return records


def parse_csv_line(line: str, min_fields: int) -> list[str] | None:
    line = line.strip()
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < min_fields:
        raise ValueError(f"Expected at least {min_fields} comma-separated fields, got: {line}")
    return parts


def defaultdict_list() -> defaultdict[int, list]:
    return defaultdict(list)


def read_sequence_list(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    sequence_names = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if name and not name.startswith("#"):
                sequence_names.add(name)
    return sequence_names


def filter_sequence_names(names: Iterable[str], requested: set[str] | None) -> list[str]:
    names = sorted(names)
    if requested is None:
        return names
    return [name for name in names if name in requested]
