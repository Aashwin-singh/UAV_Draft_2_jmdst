"""Convert VisDrone or UAVDT into the unified JMDST format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.data.converters import convert_uavdt, convert_visdrone
from jmdst.data.io import read_sequence


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", required=True, help="Dataset source root.")
    parser.add_argument("--output-root", required=True, help="Unified output root.")
    parser.add_argument("--split", required=True, help="Output split name, e.g. train/val/test.")
    parser.add_argument(
        "--sequence-list",
        default=None,
        help="Optional text file with one sequence name per line.",
    )
    parser.add_argument(
        "--collapse-to-vehicle",
        action="store_true",
        help="Collapse all vehicle subclasses to class 0/car.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the unified dataset instead of referencing source paths.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    visdrone = subparsers.add_parser("visdrone", help="Convert VisDrone2019-MOT.")
    _add_common_args(visdrone)
    visdrone.add_argument(
        "--source-split",
        default=None,
        help=(
            "Official VisDrone folder to read from (train/val/test-dev). "
            "Defaults to --split. Set this when re-splitting official data "
            "into a different train/val/test assignment than the official one."
        ),
    )

    uavdt = subparsers.add_parser("uavdt", help="Convert UAVDT.")
    _add_common_args(uavdt)
    uavdt.add_argument(
        "--all-vehicle-classes",
        action="store_true",
        help="Keep UAVDT truck/bus labels when present. Default follows the paper: car only.",
    )
    return parser


def summarize(sequence_dirs: list[Path]) -> None:
    total_frames = 0
    total_objects = 0
    for sequence_dir in sequence_dirs:
        info, records = read_sequence(sequence_dir)
        frame_objects = sum(len(record.objects) for record in records)
        total_frames += len(records)
        total_objects += frame_objects
        print(
            f"{info.dataset}/{info.split}/{info.sequence}: "
            f"{len(records)} frames, {frame_objects} objects -> {sequence_dir}"
        )
    print(
        f"Converted {len(sequence_dirs)} sequences, "
        f"{total_frames} frames, {total_objects} objects."
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset == "visdrone":
        sequence_dirs = convert_visdrone(
            source_root=args.source_root,
            output_root=args.output_root,
            split=args.split,
            sequence_list=args.sequence_list,
            collapse_to_vehicle=args.collapse_to_vehicle,
            copy_images=args.copy_images,
            source_split=args.source_split,
        )
    elif args.dataset == "uavdt":
        sequence_dirs = convert_uavdt(
            source_root=args.source_root,
            output_root=args.output_root,
            split=args.split,
            sequence_list=args.sequence_list,
            collapse_to_vehicle=args.collapse_to_vehicle,
            car_only=not args.all_vehicle_classes,
            copy_images=args.copy_images,
        )
    else:
        raise ValueError(args.dataset)

    summarize(sequence_dirs)


if __name__ == "__main__":
    main()
