"""Read and write the unified JMDST annotation format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import FrameRecord, SequenceInfo


SEQINFO_NAME = "seqinfo.json"
ANNOTATIONS_NAME = "annotations.jsonl"


def write_sequence(
    sequence_dir: str | Path,
    info: SequenceInfo,
    records: Iterable[FrameRecord],
) -> None:
    """Write one converted sequence to disk."""

    sequence_path = Path(sequence_dir)
    sequence_path.mkdir(parents=True, exist_ok=True)

    with (sequence_path / SEQINFO_NAME).open("w", encoding="utf-8") as handle:
        json.dump(info.to_dict(), handle, indent=2)
        handle.write("\n")

    with (sequence_path / ANNOTATIONS_NAME).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), separators=(",", ":")))
            handle.write("\n")


def read_sequence(sequence_dir: str | Path) -> tuple[SequenceInfo, list[FrameRecord]]:
    """Read one converted sequence from disk."""

    sequence_path = Path(sequence_dir)
    with (sequence_path / SEQINFO_NAME).open("r", encoding="utf-8") as handle:
        info = SequenceInfo.from_dict(json.load(handle))

    records: list[FrameRecord] = []
    with (sequence_path / ANNOTATIONS_NAME).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(FrameRecord.from_dict(json.loads(line)))
    return info, records


def iter_sequence_dirs(
    unified_root: str | Path,
    dataset: str | None = None,
    split: str | None = None,
) -> list[Path]:
    """Return converted sequence directories under the unified dataset root."""

    root = Path(unified_root)
    if not root.exists():
        return []

    pattern_parts = []
    pattern_parts.append(dataset if dataset else "*")
    pattern_parts.append(split if split else "*")
    pattern_parts.append("*")

    sequence_dirs = []
    for path in root.glob("/".join(pattern_parts)):
        if (path / SEQINFO_NAME).is_file() and (path / ANNOTATIONS_NAME).is_file():
            sequence_dirs.append(path)
    return sorted(sequence_dirs)


def resolve_image_path(sequence_dir: str | Path, image_path: str) -> Path:
    """Resolve an image path stored in unified annotations."""

    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(sequence_dir) / path
