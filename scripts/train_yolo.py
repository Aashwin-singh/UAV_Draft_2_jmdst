"""Train YOLOv11 on the exported unified dataset.

Per the paper (Sec. 2.5), YOLOv11 is trained independently with its own
default parameters -- by default this script does not override ultralytics'
optimizer/LR/augmentation settings, only the run-level knobs.

``--strong-aug`` is a deliberate deviation from the paper's defaults, added
to fight the overfitting observed in the first full run (val mAP peaked
~epoch 2 then declined; 44% detector FP on an out-of-distribution UAVDT val
sequence). It boosts augmentation diversity to improve generalization on the
limited, video-correlated training data. Pair it with a low ``--patience``
so training stops once val stops improving.

Usage (smoke test, ~5% of training data, 2 epochs):
    python scripts/train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt \
        --epochs 2 --fraction 0.05 --name smoke_test

Usage (faithful full run, paper defaults):
    python scripts/train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt \
        --epochs 100 --name full_run

Usage (anti-overfit retrain: stronger aug + early stopping):
    python scripts/train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt \
        --epochs 150 --patience 20 --strong-aug --name retrain_aug
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
    parser.add_argument(
        "--strong-aug",
        action="store_true",
        help="Anti-overfit augmentation preset (deviates from paper defaults). See module docstring.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


# Anti-overfit augmentation preset. ultralytics defaults leave rotation,
# mixup and copy_paste off; enabling them (plus a slightly wider scale range
# and later mosaic close) adds the pose/scale/color diversity the correlated
# video training data lacks, targeting the OOD generalization failure.
STRONG_AUG = {
    "degrees": 10.0,       # UAV footage has in-plane rotation; default 0.0
    "translate": 0.15,     # default 0.1
    "scale": 0.6,          # default 0.5
    "shear": 2.0,          # default 0.0
    "mixup": 0.15,         # default 0.0
    "copy_paste": 0.15,    # default 0.0
    "hsv_h": 0.02,         # default 0.015
    "hsv_s": 0.7,          # default 0.7
    "hsv_v": 0.5,          # default 0.4 (brightness variation for lighting shift)
    "close_mosaic": 15,    # default 10; keep mosaic on a bit longer
}


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    extra = STRONG_AUG if args.strong_aug else {}
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
        **extra,
    )


if __name__ == "__main__":
    main()
