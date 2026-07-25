"""Run YOLOv11 inference with the paper's post-processing settings (Sec. 3.3 / A.8):
NMS IoU threshold 0.2, confidence threshold 0.55.

Usage:
    python scripts/infer_yolo.py --weights outputs/yolo_runs/full_run/weights/best.pt \
        --source data/yolo/images/test --output outputs/yolo_inference
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# Paper Sec. 3.3 / PROJECT_CONTEXT.md A.8.
PAPER_NMS_IOU = 0.2
PAPER_CONF_THRESHOLD = 0.55


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image file, directory, or glob.")
    parser.add_argument("--output", default="outputs/yolo_inference")
    parser.add_argument("--conf", type=float, default=PAPER_CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=PAPER_NMS_IOU)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-txt", action="store_true", help="Also save YOLO-format label txt files.")
    parser.add_argument("--no-save-images", action="store_true", help="Skip saving annotated images.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        # Absolute path: ultralytics nests relative --project paths under its
        # own runs_dir/task rather than using the given path directly.
        project=str(Path(args.output).resolve()),
        name="predict",
        save=not args.no_save_images,
        save_txt=args.save_txt,
        exist_ok=True,
    )
    total_detections = sum(len(r.boxes) for r in results)
    print(f"Ran inference on {len(results)} images, {total_detections} detections "
          f"(conf>={args.conf}, NMS IoU<={args.iou}).")


if __name__ == "__main__":
    main()
