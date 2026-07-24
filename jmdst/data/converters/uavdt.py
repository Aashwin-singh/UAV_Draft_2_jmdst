"""UAVDT to unified JMDST conversion."""

from __future__ import annotations

from pathlib import Path

from jmdst.data.io import write_sequence
from jmdst.data.schema import (
    CLASS_NAMES,
    UAVDT_CLASS_MAP,
    ObjectAnnotation,
    SequenceInfo,
    map_class_name,
)

from .common import (
    clip_bbox_xywh,
    defaultdict_list,
    filter_sequence_names,
    group_records_by_frame,
    image_size,
    list_images,
    parse_csv_line,
    read_sequence_list,
)


def _candidate_image_roots(source_root: Path) -> list[Path]:
    return [
        source_root / "UAV-benchmark-M",
        source_root / "UAV-benchmark-MOTD_v1.0" / "UAV-benchmark-M",
        source_root / "sequences",
        source_root,
    ]


def _candidate_gt_dirs(source_root: Path) -> list[Path]:
    return [
        source_root / "GT",
        source_root / "UAV-benchmark-MOTD_v1.0" / "GT",
        source_root / "annotations",
        source_root,
    ]


def _find_uavdt_layout(source_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Find sequence image folders and GT files for common UAVDT layouts."""

    image_dirs: dict[str, Path] = {}
    for root in _candidate_image_roots(source_root):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.upper().startswith("M"):
                if list_images(child):
                    image_dirs[child.name] = child

    gt_files: dict[str, Path] = {}
    for gt_dir in _candidate_gt_dirs(source_root):
        if not gt_dir.is_dir():
            continue
        for path in gt_dir.glob("*.txt"):
            name = path.stem
            for suffix in ("_gt_whole", "_gt", "_GT"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
            if name.upper().startswith("M"):
                gt_files[name] = path

    if not image_dirs or not gt_files:
        raise FileNotFoundError(
            "Could not find UAVDT layout. Expected image folders like "
            "UAV-benchmark-M/M0101 and GT files like GT/M0101_gt_whole.txt."
        )
    return image_dirs, gt_files


def _read_uavdt_annotations(
    annotation_path: Path,
    collapse_to_vehicle: bool,
    car_only: bool,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[int, list[ObjectAnnotation]]:
    annotations_by_frame = defaultdict_list()
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = parse_csv_line(line, min_fields=8)
            if parts is None:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])

            out_of_view = int(float(parts[6])) if len(parts) > 6 else None
            occlusion = int(float(parts[7])) if len(parts) > 7 else None
            source_class_id = int(float(parts[8])) if len(parts) > 8 else 1

            class_name = UAVDT_CLASS_MAP.get(source_class_id, "car")
            if car_only and class_name != "car":
                continue
            if width <= 0 or height <= 0 or track_id < 0:
                continue

            bbox_xywh = clip_bbox_xywh([left, top, width, height], image_width, image_height)
            if bbox_xywh is None:
                continue

            class_id, unified_name = map_class_name(class_name, collapse_to_vehicle)
            visibility = None
            if out_of_view is not None:
                visibility = 0.0 if out_of_view else 1.0

            annotations_by_frame[frame_id].append(
                ObjectAnnotation(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=unified_name,
                    bbox_xywh=bbox_xywh,
                    visibility=visibility,
                    source_class_id=source_class_id,
                    source_class_name=class_name,
                    occlusion=occlusion,
                    attributes={"out_of_view": out_of_view},
                )
            )
    return dict(annotations_by_frame)


def convert_uavdt(
    source_root: str | Path,
    output_root: str | Path,
    split: str,
    sequence_list: str | Path | None = None,
    collapse_to_vehicle: bool = False,
    car_only: bool = True,
    copy_images: bool = False,
) -> list[Path]:
    """Convert UAVDT sequences into the unified JMDST format."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    image_dirs, gt_files = _find_uavdt_layout(source_root)

    requested_sequences = read_sequence_list(sequence_list)
    sequence_names = filter_sequence_names(set(image_dirs) & set(gt_files), requested_sequences)

    output_dirs: list[Path] = []
    for sequence_name in sequence_names:
        image_dir = image_dirs[sequence_name]
        annotation_path = gt_files[sequence_name]
        images_by_frame = list_images(image_dir)
        first_image = next(iter(images_by_frame.values()), None)
        width, height = image_size(first_image) if first_image else (None, None)
        annotations_by_frame = _read_uavdt_annotations(
            annotation_path=annotation_path,
            collapse_to_vehicle=collapse_to_vehicle,
            car_only=car_only,
            image_width=width,
            image_height=height,
        )

        sequence_dir = output_root / "uavdt" / split / sequence_name
        records = group_records_by_frame(
            images_by_frame=images_by_frame,
            annotations_by_frame=annotations_by_frame,
            output_sequence_dir=sequence_dir,
            copy_images=copy_images,
        )

        info = SequenceInfo(
            dataset="uavdt",
            split=split,
            sequence=sequence_name,
            frame_count=len(records),
            image_width=width,
            image_height=height,
            source_root=source_root.resolve().as_posix(),
            source_annotation_path=annotation_path.resolve().as_posix(),
            class_names=list(CLASS_NAMES),
        )
        write_sequence(sequence_dir, info, records)
        output_dirs.append(sequence_dir)

    return output_dirs
