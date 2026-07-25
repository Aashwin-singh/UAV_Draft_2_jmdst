"""Extract FELNet embedding sequences over the unified dataset (Phase 5).

Runs a trained FELNet over every unified sequence and saves per-sequence .npz
files of per-object embeddings (see jmdst.data.features). These per-target
embedding sequences are the training input for MSFP (Phase 6).

Usage:
    python scripts/extract_features.py \
        --checkpoint outputs/felnet_runs/full_run/best.pt \
        --unified-root data/unified --output-root data/features \
        --splits train val test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.data.features import extract_sequence_features, group_by_track, save_features
from jmdst.data.io import iter_sequence_dirs
from jmdst.models import FELNet, FELNetConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--output-root", default="data/features")
    parser.add_argument("--datasets", nargs="+", default=["visdrone", "uavdt"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fields = set(FELNetConfig.__dataclass_fields__)
    model = FELNet(FELNetConfig(**{k: v for k, v in checkpoint["config"].items() if k in fields}))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"Loaded FELNet (epoch {checkpoint.get('epoch')}) on {device}; embedding_dim={model.config.embedding_dim}")

    output_root = Path(args.output_root)
    total_sequences = 0
    total_objects = 0
    total_tracks = 0
    start = time.perf_counter()

    for dataset in args.datasets:
        for split in args.splits:
            sequence_dirs = iter_sequence_dirs(args.unified_root, dataset=dataset, split=split)
            if not sequence_dirs:
                continue
            for sequence_dir in sequence_dirs:
                features = extract_sequence_features(
                    model, sequence_dir, device=device, confidence_threshold=args.confidence_threshold
                )
                out_path = output_root / dataset / split / f"{sequence_dir.name}.npz"
                save_features(out_path, features)

                n_obj = features["track_ids"].shape[0]
                n_trk = len(group_by_track(features))
                total_sequences += 1
                total_objects += n_obj
                total_tracks += n_trk
                elapsed = time.perf_counter() - start
                print(
                    f"[{total_sequences:>3}] {dataset}/{split}/{sequence_dir.name}: "
                    f"{n_obj} objects, {n_trk} tracks -> {out_path}  ({elapsed:.0f}s elapsed)"
                )

    print(
        f"\nDone: {total_sequences} sequences, {total_objects} object embeddings, "
        f"{total_tracks} track sequences, {time.perf_counter()-start:.0f}s total."
    )


if __name__ == "__main__":
    main()
