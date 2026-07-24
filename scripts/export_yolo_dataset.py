"""Export the unified JMDST dataset into YOLO training format.

Usage:
    python scripts/export_yolo_dataset.py --unified-root data/unified --output-root data/yolo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.data.yolo_export import export_yolo_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--output-root", default="data/yolo")
    parser.add_argument("--datasets", nargs="+", default=["visdrone", "uavdt"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = export_yolo_dataset(
        unified_root=args.unified_root,
        output_root=args.output_root,
        datasets=args.datasets,
        splits=args.splits,
    )
    print(f"Images written: {stats.images_written}")
    print(f"Labels written: {stats.labels_written} ({stats.empty_labels} empty/background)")
    print(f"Objects written: {stats.objects_written}")
    for split, count in stats.per_split_images.items():
        print(f"  {split}: {count} images")
    if stats.skipped_missing_image:
        print(f"WARNING: {len(stats.skipped_missing_image)} images were missing and skipped.")
    print(f"dataset.yaml written under {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
