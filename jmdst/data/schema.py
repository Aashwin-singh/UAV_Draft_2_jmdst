"""Shared annotation schema for VisDrone, UAVDT, and JMDST dataloaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CLASS_NAMES = ("car", "van", "truck", "bus")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

VISDRONE_CLASS_MAP = {
    4: "car",
    5: "van",
    6: "truck",
    9: "bus",
}

VISDRONE_SOURCE_NAMES = {
    0: "ignored_regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

UAVDT_CLASS_MAP = {
    1: "car",
    2: "truck",
    3: "bus",
}


@dataclass(slots=True)
class ObjectAnnotation:
    """One object annotation in a video frame.

    The unified bbox convention is xywh in original image pixel coordinates:
    [left, top, width, height].
    """

    track_id: int
    class_id: int
    class_name: str
    bbox_xywh: list[float]
    confidence: float | None = None
    visibility: float | None = None
    source_class_id: int | None = None
    source_class_name: str | None = None
    truncation: int | None = None
    occlusion: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_xywh": [float(v) for v in self.bbox_xywh],
            "confidence": self.confidence,
            "visibility": self.visibility,
            "source_class_id": self.source_class_id,
            "source_class_name": self.source_class_name,
            "truncation": self.truncation,
            "occlusion": self.occlusion,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectAnnotation":
        return cls(
            track_id=int(data["track_id"]),
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            bbox_xywh=[float(v) for v in data["bbox_xywh"]],
            confidence=data.get("confidence"),
            visibility=data.get("visibility"),
            source_class_id=data.get("source_class_id"),
            source_class_name=data.get("source_class_name"),
            truncation=data.get("truncation"),
            occlusion=data.get("occlusion"),
            attributes=dict(data.get("attributes") or {}),
        )


@dataclass(slots=True)
class FrameRecord:
    """All annotations for one frame."""

    frame_id: int
    image_path: str
    objects: list[ObjectAnnotation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "objects": [obj.to_dict() for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameRecord":
        return cls(
            frame_id=int(data["frame_id"]),
            image_path=str(data["image_path"]),
            objects=[ObjectAnnotation.from_dict(obj) for obj in data.get("objects", [])],
        )


@dataclass(slots=True)
class SequenceInfo:
    """Metadata for a converted sequence."""

    dataset: str
    split: str
    sequence: str
    frame_count: int
    image_width: int | None = None
    image_height: int | None = None
    source_root: str | None = None
    source_annotation_path: str | None = None
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "sequence": self.sequence,
            "frame_count": self.frame_count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "source_root": self.source_root,
            "source_annotation_path": self.source_annotation_path,
            "class_names": self.class_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SequenceInfo":
        return cls(
            dataset=str(data["dataset"]),
            split=str(data["split"]),
            sequence=str(data["sequence"]),
            frame_count=int(data["frame_count"]),
            image_width=data.get("image_width"),
            image_height=data.get("image_height"),
            source_root=data.get("source_root"),
            source_annotation_path=data.get("source_annotation_path"),
            class_names=list(data.get("class_names") or CLASS_NAMES),
        )


def map_class_name(name: str, collapse_to_vehicle: bool = False) -> tuple[int, str]:
    """Return unified class id/name, optionally collapsing all vehicles to car."""

    if collapse_to_vehicle:
        return CLASS_TO_ID["car"], "car"
    if name not in CLASS_TO_ID:
        raise ValueError(f"Unsupported unified class name: {name}")
    return CLASS_TO_ID[name], name


def normalize_path(path: str | Path) -> str:
    """Store paths with forward slashes for stable JSON on Windows/Linux."""

    return Path(path).as_posix()
