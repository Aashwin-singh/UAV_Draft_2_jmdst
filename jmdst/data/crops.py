"""Small-sized image crop utilities for FELNet.

The paper crops each target-centered search region, resizes it to 64x64, and
uses a 2x2 grid of 32x32 anchors for FELNet overlap/confidence targets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .schema import ObjectAnnotation


BBox = Sequence[float]


@dataclass(slots=True)
class CropConfig:
    output_size: int = 64
    context_width_weight: float = 1.3
    context_height_weight: float = 0.3
    asymmetric_ratio_threshold: float = 2.5
    asymmetric_policy: str = "paper_min"
    random_shift_px: float = 0.0
    positive_anchor_max_attempts: int = 16
    pad_color: tuple[int, int, int] = (114, 114, 114)


@dataclass(slots=True)
class CropWindow:
    """Crop window in original image coordinates before clipping/padding."""

    left: float
    top: float
    side: float
    output_size: int = 64

    @property
    def xywh(self) -> list[float]:
        return [self.left, self.top, self.side, self.side]

    @property
    def xyxy(self) -> list[float]:
        return [self.left, self.top, self.left + self.side, self.top + self.side]


def xywh_to_xyxy(bbox: BBox) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox: BBox) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1, y1, x2 - x1, y2 - y1]


def paper_search_side(
    bbox_width: float,
    bbox_height: float,
    config: CropConfig | None = None,
) -> float:
    """Compute the target-centered square search side from the paper."""

    config = config or CropConfig()
    width = float(bbox_width)
    height = float(bbox_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox size: width={width}, height={height}")

    base = math.sqrt(
        (config.context_width_weight * width + config.context_height_weight * height)
        * (config.context_width_weight * height + config.context_height_weight * width)
    )

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio < config.asymmetric_ratio_threshold:
        return base

    paper_min = min(1.9 * width, 1.9 * height, base)
    if config.asymmetric_policy == "paper_min":
        return paper_min
    if config.asymmetric_policy == "base":
        return base
    if config.asymmetric_policy == "ensure_target":
        return max(max(width, height), paper_min)
    raise ValueError(
        "asymmetric_policy must be one of: paper_min, base, ensure_target"
    )


def compute_crop_window(
    bbox_xywh: BBox,
    config: CropConfig | None = None,
    rng: random.Random | None = None,
) -> CropWindow:
    """Return the unclipped target-centered SSI crop window."""

    config = config or CropConfig()
    x, y, width, height = [float(v) for v in bbox_xywh]
    side = paper_search_side(width, height, config)
    center_x = x + width / 2.0
    center_y = y + height / 2.0

    if config.random_shift_px > 0:
        rng = rng or random
        center_x += rng.uniform(-config.random_shift_px, config.random_shift_px)
        center_y += rng.uniform(-config.random_shift_px, config.random_shift_px)

    return CropWindow(
        left=center_x - side / 2.0,
        top=center_y - side / 2.0,
        side=side,
        output_size=config.output_size,
    )


def crop_ssi(
    image: Image.Image,
    bbox_xywh: BBox,
    config: CropConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[Image.Image, CropWindow]:
    """Crop a target SSI and resize it to config.output_size x config.output_size."""

    config = config or CropConfig()
    image = image.convert("RGB")
    window = compute_crop_window(bbox_xywh, config=config, rng=rng)
    x1, y1, x2, y2 = window.xyxy

    # Integer crop bounds are used only for pixels. The returned CropWindow keeps
    # the floating-point geometry used for label generation.
    ix1 = math.floor(x1)
    iy1 = math.floor(y1)
    ix2 = math.ceil(x2)
    iy2 = math.ceil(y2)
    crop_w = max(1, ix2 - ix1)
    crop_h = max(1, iy2 - iy1)

    canvas = Image.new("RGB", (crop_w, crop_h), color=config.pad_color)
    src_x1 = max(0, ix1)
    src_y1 = max(0, iy1)
    src_x2 = min(image.width, ix2)
    src_y2 = min(image.height, iy2)
    if src_x2 > src_x1 and src_y2 > src_y1:
        patch = image.crop((src_x1, src_y1, src_x2, src_y2))
        canvas.paste(patch, (src_x1 - ix1, src_y1 - iy1))

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return canvas.resize((config.output_size, config.output_size), resampling), window


def bbox_image_to_ssi(bbox_xywh: BBox, window: CropWindow) -> list[float]:
    """Map an original-image bbox into 64x64 SSI pixel coordinates."""

    x, y, width, height = [float(v) for v in bbox_xywh]
    scale = window.output_size / window.side
    return [
        (x - window.left) * scale,
        (y - window.top) * scale,
        width * scale,
        height * scale,
    ]


def bbox_ssi_to_image(bbox_xywh: BBox, window: CropWindow) -> list[float]:
    """Map an SSI bbox back into original-image coordinates."""

    x, y, width, height = [float(v) for v in bbox_xywh]
    scale = window.side / window.output_size
    return [
        window.left + x * scale,
        window.top + y * scale,
        width * scale,
        height * scale,
    ]


def overlap_vector(anchor_xywh: BBox, object_xywh: BBox) -> list[float]:
    """Eq. 1 overlap vector o(B_S, B_H) for xywh boxes."""

    l1, u1, w1, h1 = [float(v) for v in anchor_xywh]
    l2, u2, w2, h2 = [float(v) for v in object_xywh]
    return [
        max(0.0, l1 + w1 - l2),
        max(0.0, l2 + w2 - l1),
        max(0.0, u1 + h1 - u2),
        max(0.0, u2 + h2 - u1),
    ]


def overlap_to_bbox(anchor_xywh: BBox, overlap: BBox) -> list[float]:
    """Eq. 2 inverse conversion from overlap vector to object bbox."""

    l1, u1, w1, h1 = [float(v) for v in anchor_xywh]
    o1, o2, o3, o4 = [float(v) for v in overlap]
    return [
        l1 + w1 - o1,
        u1 + h1 - o3,
        o1 + o2 - w1,
        o3 + o4 - h1,
    ]


def anchor_boxes(output_size: int = 64, grid_size: int = 2) -> list[list[float]]:
    """Return FELNet anchor boxes as xywh boxes in SSI coordinates."""

    cell = output_size / grid_size
    anchors = []
    for row in range(grid_size):
        for col in range(grid_size):
            anchors.append([col * cell, row * cell, cell, cell])
    return anchors


def point_inside_bbox(point: tuple[float, float], bbox_xywh: BBox) -> bool:
    x, y, width, height = [float(v) for v in bbox_xywh]
    px, py = point
    return x <= px <= x + width and y <= py <= y + height


def _anchor_deviation(anchor_xywh: BBox, overlap: BBox) -> float:
    _, _, width, height = [float(v) for v in anchor_xywh]
    target = np.array([width, width, height, height], dtype=np.float32)
    return float(np.linalg.norm(np.array(overlap, dtype=np.float32) - target))


def make_felnet_targets(
    objects: Iterable[ObjectAnnotation],
    window: CropWindow,
    output_size: int = 64,
    grid_size: int = 2,
) -> dict[str, np.ndarray]:
    """Build FELNet overlap/confidence labels for the four SSI anchors."""

    anchors = anchor_boxes(output_size=output_size, grid_size=grid_size)
    object_list = list(objects)
    overlaps = np.zeros((len(anchors), 4), dtype=np.float32)
    confidences = np.zeros((len(anchors),), dtype=np.float32)
    class_ids = np.full((len(anchors),), -1, dtype=np.int64)
    track_ids = np.full((len(anchors),), -1, dtype=np.int64)
    bboxes = np.zeros((len(anchors), 4), dtype=np.float32)

    ssi_bboxes = [
        (obj, bbox_image_to_ssi(obj.bbox_xywh, window))
        for obj in object_list
    ]

    for anchor_index, anchor in enumerate(anchors):
        ax, ay, aw, ah = anchor
        center = (ax + aw / 2.0, ay + ah / 2.0)
        best = None
        best_deviation = float("inf")
        for obj, bbox_ssi in ssi_bboxes:
            if not point_inside_bbox(center, bbox_ssi):
                continue
            overlap = overlap_vector(anchor, bbox_ssi)
            deviation = _anchor_deviation(anchor, overlap)
            if deviation < best_deviation:
                best = (obj, bbox_ssi, overlap)
                best_deviation = deviation

        if best is None:
            continue
        obj, bbox_ssi, overlap = best
        overlaps[anchor_index] = np.array(overlap, dtype=np.float32)
        confidences[anchor_index] = 1.0
        class_ids[anchor_index] = obj.class_id
        track_ids[anchor_index] = obj.track_id
        bboxes[anchor_index] = np.array(bbox_ssi, dtype=np.float32)

    return {
        "anchors_xywh": np.array(anchors, dtype=np.float32),
        "overlaps": overlaps,
        "confidences": confidences,
        "class_ids": class_ids,
        "track_ids": track_ids,
        "bboxes_xywh": bboxes,
    }


def selected_track_has_positive_anchor(targets: dict[str, np.ndarray], track_id: int) -> bool:
    """Return True if a target track owns at least one positive FELNet anchor."""

    positive = targets["confidences"] > 0
    if not np.any(positive):
        return False
    return bool(np.any(targets["track_ids"][positive] == int(track_id)))


def crop_ssi_with_targets(
    image: Image.Image,
    center_object: ObjectAnnotation,
    objects: Iterable[ObjectAnnotation],
    config: CropConfig | None = None,
    rng: random.Random | None = None,
    require_positive_anchor: bool = False,
) -> tuple[Image.Image, CropWindow, dict[str, np.ndarray]]:
    """Crop an SSI and build FELNet targets, retrying shifts if requested."""

    config = config or CropConfig()
    object_list = list(objects)
    attempts = max(1, config.positive_anchor_max_attempts if require_positive_anchor else 1)
    last_result = None
    for _ in range(attempts):
        ssi, window = crop_ssi(image, center_object.bbox_xywh, config=config, rng=rng)
        targets = make_felnet_targets(
            object_list,
            window,
            output_size=config.output_size,
            grid_size=2,
        )
        last_result = (ssi, window, targets)
        if not require_positive_anchor or selected_track_has_positive_anchor(
            targets, center_object.track_id
        ):
            return last_result

    if last_result is None:
        raise RuntimeError("Failed to create SSI crop.")
    return last_result
