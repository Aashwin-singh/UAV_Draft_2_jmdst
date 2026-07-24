"""VisDrone2019-MOT to unified JMDST conversion."""

from __future__ import annotations

from pathlib import Path

from jmdst.data.io import write_sequence
from jmdst.data.schema import (
    CLASS_NAMES,
    VISDRONE_CLASS_MAP,
    VISDRONE_SOURCE_NAMES,
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


def _find_visdrone_layout(source_root: Path, split: str) -> tuple[Path, Path]:
    """Find sequences and annotations folders for common VisDrone layouts."""

    candidates = [
        source_root,
        source_root / f"VisDrone2019-MOT-{split}",
        source_root / f"VisDrone2019-MOT-{split.capitalize()}",
    ]
    for base in candidates:
        sequences = base / "sequences"
        annotations = base / "annotations"
        if sequences.is_dir() and annotations.is_dir():
            return sequences, annotations
    raise FileNotFoundError(
        "Could not find VisDrone layout. Expected folders like "
        f"{source_root}/VisDrone2019-MOT-{split}/sequences and annotations."
    )


def _read_visdrone_annotations(
    annotation_path: Path,
    collapse_to_vehicle: bool,
    include_ignored: bool = False,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[int, list[ObjectAnnotation]]:
    annotations_by_frame = defaultdict_list()
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = parse_csv_line(line, min_fields=10)
            if parts is None:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            confidence = float(parts[6])
            source_class_id = int(float(parts[7]))
            truncation = int(float(parts[8]))
            occlusion = int(float(parts[9]))

            if source_class_id == 0 and include_ignored:
                continue
            class_name = VISDRONE_CLASS_MAP.get(source_class_id)
            if class_name is None:
                continue
            if width <= 0 or height <= 0 or track_id < 0:
                continue

            bbox_xywh = clip_bbox_xywh([left, top, width, height], image_width, image_height)
            if bbox_xywh is None:
                continue

            class_id, unified_name = map_class_name(class_name, collapse_to_vehicle)
            annotations_by_frame[frame_id].append(
                ObjectAnnotation(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=unified_name,
                    bbox_xywh=bbox_xywh,
                    confidence=confidence,
                    source_class_id=source_class_id,
                    source_class_name=VISDRONE_SOURCE_NAMES.get(source_class_id),
                    truncation=truncation,
                    occlusion=occlusion,
                )
            )
    return dict(annotations_by_frame)


def convert_visdrone(
    source_root: str | Path,
    output_root: str | Path,
    split: str,
    sequence_list: str | Path | None = None,
    collapse_to_vehicle: bool = False,
    copy_images: bool = False,
) -> list[Path]:
    """Convert a VisDrone split into the unified JMDST format."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    sequences_dir, annotations_dir = _find_visdrone_layout(source_root, split)

    requested_sequences = read_sequence_list(sequence_list)
    annotation_paths = {path.stem: path for path in annotations_dir.glob("*.txt")}
    sequence_names = filter_sequence_names(annotation_paths.keys(), requested_sequences)

    output_dirs: list[Path] = []
    for sequence_name in sequence_names:
        image_dir = sequences_dir / sequence_name
        if not image_dir.is_dir():
            continue

        annotation_path = annotation_paths[sequence_name]
        images_by_frame = list_images(image_dir)
        first_image = next(iter(images_by_frame.values()), None)
        width, height = image_size(first_image) if first_image else (None, None)
        annotations_by_frame = _read_visdrone_annotations(
            annotation_path=annotation_path,
            collapse_to_vehicle=collapse_to_vehicle,
            image_width=width,
            image_height=height,
        )

        sequence_dir = output_root / "visdrone" / split / sequence_name
        records = group_records_by_frame(
            images_by_frame=images_by_frame,
            annotations_by_frame=annotations_by_frame,
            output_sequence_dir=sequence_dir,
            copy_images=copy_images,
        )

        info = SequenceInfo(
            dataset="visdrone",
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
