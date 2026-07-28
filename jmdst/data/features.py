"""Extract and store per-target FELNet embedding sequences (Phase 5).

Runs a trained FELNet over unified sequences and, for every ground-truth
object in every frame, records the target's 16-D embedding. Grouped by track
id and ordered by frame, these become the per-target embedding *sequences*
that MSFP (Phase 6) learns to predict.

Each object's embedding is taken from a GT-box-centered 64x64 crop (no
augmentation, deterministic) at the anchor chosen by the paper's selection
rule (select_anchor_output) using the GT box as the reference -- i.e. exactly
the embedding the tracker would consume for this target, so MSFP trains on
the same distribution it will see at inference.

Per-sequence storage is a flat .npz:
    track_ids   (N,)    int64
    frame_ids   (N,)    int64
    embeddings  (N, D)  float32
    boxes_xywh  (N, 4)  float32   (the GT box each embedding came from)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .crops import CropConfig, anchor_boxes, bbox_image_to_ssi, crop_ssi
from .datasets import _to_tensor
from .io import read_sequence, resolve_image_path
from .schema import SequenceInfo

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


def extract_sequence_features(
    model: Any,
    sequence_dir: str | Path,
    device: str = "cpu",
    confidence_threshold: float = 0.9,
) -> dict[str, np.ndarray]:
    """Extract per-object embeddings for one unified sequence.

    Returns flat arrays (track_ids, frame_ids, embeddings, boxes_xywh); use
    ``group_by_track`` to turn them into per-target time series.
    """

    from jmdst.models import select_anchor_output  # local import avoids cycle

    from PIL import Image

    sequence_dir = Path(sequence_dir)
    _info, records = read_sequence(sequence_dir)
    crop_config = CropConfig(random_shift_px=0.0)
    anchors = torch.tensor(anchor_boxes(64, 2), dtype=torch.float32, device=device)

    track_ids: list[int] = []
    frame_ids: list[int] = []
    embeddings: list[np.ndarray] = []
    boxes: list[list[float]] = []

    model.eval()
    for record in records:
        if not record.objects:
            continue

        image = Image.open(resolve_image_path(sequence_dir, record.image_path)).convert("RGB")

        ssis = []
        refs = []
        metas = []
        for obj in record.objects:
            ssi, window = crop_ssi(image, obj.bbox_xywh, config=crop_config)
            ref_box_ssi = bbox_image_to_ssi(obj.bbox_xywh, window)
            ssis.append(_to_tensor(ssi))
            refs.append(ref_box_ssi)
            metas.append((int(obj.track_id), int(record.frame_id), list(obj.bbox_xywh)))

        x = torch.stack(ssis, dim=0).to(device)
        ref = torch.tensor(refs, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(x)
            # Anchor selection compares against pixel-space reference overlaps.
            out["overlap"] = out["overlap"] * model.config.overlap_scale
            selected = select_anchor_output(
                out["overlap"], out["confidence"], ref, anchors, confidence_threshold
            )
            rows = torch.arange(x.shape[0], device=device)
            emb = out["embedding"][rows, selected].cpu().numpy()

        for i, (tid, fid, box) in enumerate(metas):
            track_ids.append(tid)
            frame_ids.append(fid)
            embeddings.append(emb[i])
            boxes.append(box)

    embedding_dim = model.config.embedding_dim
    return {
        "track_ids": np.asarray(track_ids, dtype=np.int64),
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "embeddings": np.asarray(embeddings, dtype=np.float32).reshape(-1, embedding_dim),
        "boxes_xywh": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
    }


def save_features(path: str | Path, features: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def group_by_track(features: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    """Group flat features into per-target, frame-ordered embedding sequences.

    Returns {track_id: {"frame_ids": (T,), "embeddings": (T, D), "boxes_xywh": (T, 4)}}
    with each target's rows sorted by frame id.
    """

    track_ids = features["track_ids"]
    frame_ids = features["frame_ids"]
    embeddings = features["embeddings"]
    boxes = features["boxes_xywh"]

    grouped: dict[int, dict[str, np.ndarray]] = {}
    for tid in np.unique(track_ids):
        mask = track_ids == tid
        order = np.argsort(frame_ids[mask], kind="stable")
        grouped[int(tid)] = {
            "frame_ids": frame_ids[mask][order],
            "embeddings": embeddings[mask][order],
            "boxes_xywh": boxes[mask][order],
        }
    return grouped
