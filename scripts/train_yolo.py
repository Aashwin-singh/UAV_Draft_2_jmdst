"""Train YOLOv11 on the exported unified dataset.

Per the paper (Sec. 2.5), YOLOv11 is trained independently with its own
default parameters -- this script does not override ultralytics' default
optimizer/LR/augmentation settings, only the run-level knobs (epochs,
image size, batch size, which pretrained checkpoint to start from).

Usage (smoke test, ~5% of training data, 2 epochs):
    python scripts/train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt \
        --epochs 2 --fraction 0.05 --name smoke_test

Usage (full run):
    python scripts/train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt \
        --epochs 100 --name full_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/yolo/dataset.yaml")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Starting checkpoint (COCO-pretrained .pt) or architecture .yaml for training from scratch.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0 for first GPU, or 'cpu'. Defaults to ultralytics auto-detect.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use per epoch (for smoke tests).")
    parser.add_argument("--project", default="outputs/yolo_runs")
    parser.add_argument("--name", default="train")
    parser.add_argument("--patience", type=int, default=100, help="Early-stopping patience (epochs without improvement).")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction,
        # Resolve to an absolute path: ultralytics nests relative --project
        # paths under its own global runs_dir/task ("runs/detect/...")
        # rather than using the given path directly.
        project=str(Path(args.project).resolve()),
        name=args.name,
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
