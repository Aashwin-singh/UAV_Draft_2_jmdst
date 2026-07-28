"""Episode sampler for FELNet training (paper Algorithm 2).

Each episode reproduces the paper's dense random sampling strategy:

    k ~ Uniform[1, k_max]
    sample N_o frames from one sequence at interval k
    crop N_s SSIs per frame from GT object labels (with augmentation)

Because the N_o frames come from a single sequence at a short interval, the
same targets recur across frames, so the N_o x N_s SSIs contain natural
positive (same-target) and negative (different-target) embedding pairs -- the
supervision the feature-similarity loss (paper Eq. 5/6) needs. Random,
independent per-frame SSIs (as in FELNetSSIDataset) would not reliably produce
positive pairs, which is why FELNet training needs this dedicated sampler.

Each __getitem__ returns one fully-cropped episode as stacked arrays, so the
DataLoader batch dimension is "episodes per step". Identity ids are local to
the episode (a sequence's track_id space is already unique within it); the
collate function offsets them so they stay unique across a multi-episode batch.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from .crops import CropConfig, crop_ssi_with_targets
from .datasets import TORCH_AVAILABLE, Dataset, _to_tensor
from .io import iter_sequence_dirs, read_sequence
from .schema import FrameRecord, ObjectAnnotation, SequenceInfo

if TORCH_AVAILABLE:
    import torch


# Horizontal-flip permutation of the 2x2 anchor grid (row-major indices):
# top-left<->top-right, bottom-left<->bottom-right.
_FLIP_ANCHOR_PERM = [1, 0, 3, 2]


def _flip_targets_horizontally(targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply a horizontal flip to FELNet anchor targets.

    Reorders the four anchors by the flip permutation and, within each
    overlap vector [o1(left), o2(right), o3(top), o4(bottom)], swaps the left
    and right overlaps. Verified in tests to keep overlap<->box consistent.
    """

    perm = _FLIP_ANCHOR_PERM
    overlaps = targets["overlaps"][perm].copy()
    # swap o1 (left) and o2 (right)
    overlaps[:, [0, 1]] = overlaps[:, [1, 0]]
    return {
        "overlaps": overlaps,
        "confidences": targets["confidences"][perm].copy(),
        "class_ids": targets["class_ids"][perm].copy(),
        "track_ids": targets["track_ids"][perm].copy(),
        "anchors_xywh": targets["anchors_xywh"],
        "bboxes_xywh": targets["bboxes_xywh"][perm].copy(),
    }


def _center_anchor_index(targets: dict[str, np.ndarray], center_track_id: int) -> int:
    """Index of the positive anchor owning the center target, or -1 if none."""

    positive = targets["confidences"] > 0
    owns = positive & (targets["track_ids"] == int(center_track_id))
    where = np.nonzero(owns)[0]
    if where.size:
        return int(where[0])
    return -1


class FELNetEpisodeDataset(Dataset):
    """Samples paper-style FELNet training episodes.

    Args:
        unified_root/dataset/split: where to read unified sequences from.
        k_max: max sampling interval (paper's k_max). k ~ Uniform[1, k_max].
        n_o: frames per episode (paper's N_o).
        n_s: SSIs cropped per frame (paper's N_s).
        length: number of episodes per epoch (episodes are random, so __len__
            just controls how many steps an epoch runs).
        crop_config: SSI crop config; defaults apply the paper's 16px train
            shift augmentation.
        horizontal_flip: apply the paper's flip augmentation (target-consistent).
        seed: base RNG seed.
    """

    def __init__(
        self,
        unified_root: str | Path,
        dataset: str | None = None,
        split: str | None = "train",
        k_max: int = 5,
        n_o: int = 8,
        n_s: int = 8,
        length: int = 2000,
        crop_config: CropConfig | None = None,
        horizontal_flip: bool = True,
        seed: int | None = None,
        overlap_scale: float = 1.0,
    ) -> None:
        # Overlap targets are divided by overlap_scale, so overlap_scale=64
        # yields ~[0,1] targets whose RMSE is comparable to the embedding and
        # confidence losses (see FELNetConfig.overlap_scale). Must match the
        # model's FELNetConfig.overlap_scale.
        self.overlap_scale = float(overlap_scale)
        self.k_max = int(k_max)
        self.n_o = int(n_o)
        self.n_s = int(n_s)
        self.length = int(length)
        self.crop_config = crop_config or CropConfig(random_shift_px=16.0)
        self.horizontal_flip = horizontal_flip
        self.base_seed = seed

        self.sequences: list[tuple[Path, SequenceInfo, list[FrameRecord]]] = []
        for sequence_dir in iter_sequence_dirs(unified_root, dataset=dataset, split=split):
            info, records = read_sequence(sequence_dir)
            usable = [rec for rec in records if rec.objects]
            if len(usable) >= 1:
                self.sequences.append((sequence_dir, info, usable))

        if not self.sequences:
            raise ValueError(
                f"No usable sequences with objects found under {unified_root} "
                f"(dataset={dataset}, split={split})."
            )

    def __len__(self) -> int:
        return self.length

    def _select_frames(self, usable: list[FrameRecord], rng: random.Random) -> list[FrameRecord]:
        n_frames = len(usable)
        k = rng.randint(1, self.k_max)
        # Shrink k if the sequence is too short to span N_o frames at interval k.
        if (self.n_o - 1) * k >= n_frames:
            k = max(1, (n_frames - 1) // max(1, self.n_o - 1))

        max_start = max(0, n_frames - 1 - (self.n_o - 1) * k)
        start = rng.randint(0, max_start) if max_start > 0 else 0

        positions = [start + i * k for i in range(self.n_o) if start + i * k < n_frames]
        return [usable[p] for p in positions]

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Per-item RNG so DataLoader workers stay reproducible yet varied.
        seed = None if self.base_seed is None else self.base_seed + index
        rng = random.Random(seed)

        sequence_dir, info, usable = rng.choice(self.sequences)
        frames = self._select_frames(usable, rng)

        ssis: list[Any] = []
        overlaps: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        center_anchor_indices: list[int] = []
        track_ids: list[int] = []

        # Cache decoded images per frame so N_s crops reuse a single load.
        from PIL import Image

        for record in frames:
            image_path = sequence_dir / record.image_path if not Path(record.image_path).is_absolute() else Path(record.image_path)
            image = Image.open(image_path).convert("RGB")

            objects = list(record.objects)
            n_take = min(self.n_s, len(objects))
            chosen = rng.sample(objects, n_take)

            for center_obj in chosen:
                ssi, _window, targets = crop_ssi_with_targets(
                    image,
                    center_obj,
                    objects,
                    config=self.crop_config,
                    rng=rng,
                    require_positive_anchor=True,
                )

                if self.horizontal_flip and rng.random() < 0.5:
                    ssi = ssi.transpose(Image.FLIP_LEFT_RIGHT)
                    targets = _flip_targets_horizontally(targets)

                anchor_index = _center_anchor_index(targets, center_obj.track_id)
                if anchor_index < 0:
                    # No positive anchor for the center target after augmentation;
                    # skip so embedding sampling always has a valid anchor.
                    continue

                ssis.append(_to_tensor(ssi))
                overlaps.append(targets["overlaps"])
                confidences.append(targets["confidences"])
                center_anchor_indices.append(anchor_index)
                track_ids.append(int(center_obj.track_id))

        # Map track_ids to contiguous local identity ids 0..K-1.
        unique_ids = {tid: i for i, tid in enumerate(sorted(set(track_ids)))}
        identity = [unique_ids[tid] for tid in track_ids]

        if self.overlap_scale != 1.0:
            overlaps = [o / self.overlap_scale for o in overlaps]

        if TORCH_AVAILABLE:
            return {
                "ssi": torch.stack(ssis, dim=0),
                "overlaps": torch.as_tensor(np.stack(overlaps), dtype=torch.float32),
                "confidences": torch.as_tensor(np.stack(confidences), dtype=torch.float32),
                "center_anchor_index": torch.as_tensor(center_anchor_indices, dtype=torch.long),
                "identity": torch.as_tensor(identity, dtype=torch.long),
            }
        return {
            "ssi": np.stack(ssis),
            "overlaps": np.stack(overlaps),
            "confidences": np.stack(confidences),
            "center_anchor_index": np.asarray(center_anchor_indices, dtype=np.int64),
            "identity": np.asarray(identity, dtype=np.int64),
        }


def collate_felnet_episodes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge episodes into one batch, keeping identity ids globally unique."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to collate FELNet episodes.")

    ssis = []
    overlaps = []
    confidences = []
    anchor_indices = []
    identities = []
    offset = 0
    for episode in batch:
        ssis.append(episode["ssi"])
        overlaps.append(episode["overlaps"])
        confidences.append(episode["confidences"])
        anchor_indices.append(episode["center_anchor_index"])
        ident = episode["identity"] + offset
        identities.append(ident)
        offset = int(ident.max().item()) + 1 if ident.numel() else offset

    return {
        "ssi": torch.cat(ssis, dim=0),
        "overlaps": torch.cat(overlaps, dim=0),
        "confidences": torch.cat(confidences, dim=0),
        "center_anchor_index": torch.cat(anchor_indices, dim=0),
        "identity": torch.cat(identities, dim=0),
    }
