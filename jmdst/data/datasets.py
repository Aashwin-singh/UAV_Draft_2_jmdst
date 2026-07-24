"""PyTorch-ready dataloaders for unified JMDST datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .crops import CropConfig, bbox_image_to_ssi, crop_ssi_with_targets, xywh_to_xyxy
from .io import iter_sequence_dirs, read_sequence, resolve_image_path
from .schema import FrameRecord, ObjectAnnotation, SequenceInfo

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on machines without torch.
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        pass

    TORCH_AVAILABLE = False


def _to_tensor(image: Image.Image) -> Any:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    if TORCH_AVAILABLE:
        return torch.from_numpy(array)
    return array


def _array(values: list, dtype: Any) -> Any:
    array = np.asarray(values, dtype=dtype)
    if TORCH_AVAILABLE:
        return torch.as_tensor(array)
    return array


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return Image.open(path).convert("RGB")


class UnifiedDetectionDataset(Dataset):
    """Frame-level detection dataset from unified annotations."""

    def __init__(
        self,
        unified_root: str | Path,
        dataset: str | None = None,
        split: str | None = None,
        include_empty: bool = True,
        return_pil: bool = False,
    ) -> None:
        self.sequence_dirs = iter_sequence_dirs(unified_root, dataset=dataset, split=split)
        self.frames: list[tuple[Path, SequenceInfo, FrameRecord]] = []
        self.return_pil = return_pil

        for sequence_dir in self.sequence_dirs:
            info, records = read_sequence(sequence_dir)
            for record in records:
                if include_empty or record.objects:
                    self.frames.append((sequence_dir, info, record))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        sequence_dir, info, record = self.frames[index]
        image = _load_image(resolve_image_path(sequence_dir, record.image_path))
        boxes_xyxy = [xywh_to_xyxy(obj.bbox_xywh) for obj in record.objects]
        labels = [obj.class_id for obj in record.objects]
        track_ids = [obj.track_id for obj in record.objects]

        target = {
            "boxes": _array(boxes_xyxy, np.float32).reshape((-1, 4)),
            "labels": _array(labels, np.int64),
            "track_ids": _array(track_ids, np.int64),
            "frame_id": record.frame_id,
            "sequence": info.sequence,
            "dataset": info.dataset,
            "image_path": str(resolve_image_path(sequence_dir, record.image_path)),
        }
        return (image if self.return_pil else _to_tensor(image)), target


class FELNetSSIDataset(Dataset):
    """SSI crop dataset for FELNet localization and feature training."""

    def __init__(
        self,
        unified_root: str | Path,
        dataset: str | None = None,
        split: str | None = None,
        crop_config: CropConfig | None = None,
        training: bool = True,
        return_pil: bool = False,
        seed: int | None = None,
        require_positive_anchor: bool | None = None,
    ) -> None:
        self.sequence_dirs = iter_sequence_dirs(unified_root, dataset=dataset, split=split)
        self.crop_config = crop_config or CropConfig(random_shift_px=16.0 if training else 0.0)
        self.training = training
        self.return_pil = return_pil
        self.rng = random.Random(seed)
        self.require_positive_anchor = training if require_positive_anchor is None else require_positive_anchor
        self.samples: list[tuple[Path, SequenceInfo, FrameRecord, ObjectAnnotation]] = []

        for sequence_dir in self.sequence_dirs:
            info, records = read_sequence(sequence_dir)
            for record in records:
                for obj in record.objects:
                    self.samples.append((sequence_dir, info, record, obj))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        sequence_dir, info, record, center_obj = self.samples[index]
        image_path = resolve_image_path(sequence_dir, record.image_path)
        image = _load_image(image_path)
        rng = self.rng if self.training else None
        ssi, window, felnet_targets = crop_ssi_with_targets(
            image,
            center_obj,
            record.objects,
            config=self.crop_config,
            rng=rng,
            require_positive_anchor=self.require_positive_anchor,
        )

        target = {
            "overlaps": _array(felnet_targets["overlaps"].tolist(), np.float32),
            "confidences": _array(felnet_targets["confidences"].tolist(), np.float32),
            "class_ids": _array(felnet_targets["class_ids"].tolist(), np.int64),
            "track_ids": _array(felnet_targets["track_ids"].tolist(), np.int64),
            "anchors_xywh": _array(felnet_targets["anchors_xywh"].tolist(), np.float32),
            "anchor_bboxes_xywh": _array(felnet_targets["bboxes_xywh"].tolist(), np.float32),
            "center_bbox_xywh": _array(center_obj.bbox_xywh, np.float32),
            "center_bbox_ssi_xywh": _array(bbox_image_to_ssi(center_obj.bbox_xywh, window), np.float32),
            "crop_xyxy": _array(window.xyxy, np.float32),
            "center_track_id": center_obj.track_id,
            "center_class_id": center_obj.class_id,
            "frame_id": record.frame_id,
            "sequence": info.sequence,
            "dataset": info.dataset,
            "image_path": str(image_path),
        }
        return (ssi if self.return_pil else _to_tensor(ssi)), target


class TrackingSequenceDataset(Dataset):
    """Frame iterator for sequence-level tracking inference/evaluation."""

    def __init__(
        self,
        unified_root: str | Path,
        dataset: str | None = None,
        split: str | None = None,
        return_pil: bool = False,
    ) -> None:
        self.sequence_dirs = iter_sequence_dirs(unified_root, dataset=dataset, split=split)
        self.index: list[tuple[Path, SequenceInfo, FrameRecord]] = []
        self.return_pil = return_pil
        for sequence_dir in self.sequence_dirs:
            info, records = read_sequence(sequence_dir)
            self.index.extend((sequence_dir, info, record) for record in records)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        sequence_dir, info, record = self.index[index]
        image_path = resolve_image_path(sequence_dir, record.image_path)
        image = _load_image(image_path)
        meta = {
            "frame_id": record.frame_id,
            "sequence": info.sequence,
            "dataset": info.dataset,
            "image_path": str(image_path),
            "objects": [obj.to_dict() for obj in record.objects],
        }
        return (image if self.return_pil else _to_tensor(image)), meta


def collate_detection(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[Any, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    if TORCH_AVAILABLE and all(hasattr(image, "shape") for image in images):
        return torch.stack(list(images), dim=0), list(targets)
    return list(images), list(targets)


def collate_felnet(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[Any, dict[str, Any]]:
    images, targets = zip(*batch)
    if not TORCH_AVAILABLE:
        return list(images), {"samples": list(targets)}

    batch_target: dict[str, Any] = {}
    tensor_keys = [
        "overlaps",
        "confidences",
        "class_ids",
        "track_ids",
        "anchors_xywh",
        "anchor_bboxes_xywh",
        "center_bbox_xywh",
        "center_bbox_ssi_xywh",
        "crop_xyxy",
    ]
    for key in tensor_keys:
        batch_target[key] = torch.stack([target[key] for target in targets], dim=0)
    for key in ["center_track_id", "center_class_id", "frame_id", "sequence", "dataset", "image_path"]:
        batch_target[key] = [target[key] for target in targets]
    return torch.stack(list(images), dim=0), batch_target


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Any | None = None,
) -> Any:
    """Build a torch DataLoader with an actionable error if torch is missing."""

    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required to build dataloaders. Install torch, then rerun."
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
