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

Usage (YOLOv11m capacity upgrade -- note the mandatory smaller ``--batch``):
    python scripts/train_yolo.py --data data/yolo_uavdt/dataset.yaml --model yolo11m.pt \
        --epochs 150 --patience 20 --batch 8 --strong-aug --name uavdt_only_m


Choosing ``--batch`` for a model size (RTX 4070 Laptop, 8 GB)
------------------------------------------------------------
The paper does not specify a YOLOv11 size. All detectors through 2026-07-28
(``full_run``, ``retrain_aug``, ``uavdt_only``, ``visdrone_only``) are
**yolo11n at batch 16**. Larger variants do not fit at that batch size on this
card. Measured peak reserved VRAM, imgsz 640, AMP on, forward+backward:

    yolo11n  batch 16   2.38 GB    <- what the existing detectors used
    yolo11s  batch 16   4.18 GB
    yolo11m  batch 16   8.19 GB    <- exceeds the whole 8 GB card: OOM
    yolo11m  batch 12   6.18 GB    <- too tight once the val pass runs
    yolo11m  batch  8   4.25 GB    <- safe; use this

Those figures exclude the TAL assigner, dataloader and validation pass, so
real usage is somewhat higher -- treat them as lower bounds, not headroom.

Dropping batch 16 -> 8 does **not** invalidate an A/B against the yolo11n
detectors: ultralytics accumulates gradients to a nominal batch of 64
(``nbs=64``), so the optimizer step is equivalent (accumulate 4 -> 8) and
weight decay is rescaled to match. Only BatchNorm statistics differ.

Cost of the upgrade, measured on this GPU: ~20.2 ms/image to train (vs
~4.9 for yolo11n at batch 16), i.e. ~2.9 min/epoch compute-only on
``data/yolo_uavdt`` (8,472 imgs) and ~8.7 min/epoch on ``data/yolo``
(25,742 imgs). Inference costs +3.3 ms/detection-frame on UAVDT and
+2.3 ms on VisDrone; at tau=3 that is a third of it per frame, since the
JMDST loop is dominated by FELNet's per-target SSI work rather than YOLO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    # Raw formatter so the VRAM table in the module docstring survives --help.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="data/yolo/dataset.yaml")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help=(
            "Starting checkpoint (COCO-pretrained .pt) or architecture .yaml for training "
            "from scratch. Default stays yolo11n.pt so the commands recorded in "
            "project_log.md keep reproducing; pass yolo11m.pt explicitly (with --batch 8)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Fits yolo11n/yolo11s on an 8 GB card. yolo11m needs --batch 8; see module docstring.",
    )
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
