"""Run the full JMDST pipeline over unified sequences (Phase 9).

For each sequence, runs Algorithm 1's dual-branch routine and writes the
confirmed-tracklet outputs in MOT-Challenge format:

    frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1

one file per sequence under <output-root>/<dataset>/<split>/<sequence>.txt.
These are the tracker results Phase 10 evaluation scores against the GT.

Usage:
    python scripts/run_jmdst.py \
        --yolo outputs/yolo_runs/full_run/weights/best.pt \
        --felnet outputs/felnet_runs/full_run/best.pt \
        --dataset uavdt --split val --tau 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from jmdst.data.io import iter_sequence_dirs, read_sequence, resolve_image_path
from jmdst.pipeline.cli import add_tracker_arguments, detector_conf, mot_lines, tracker_from_args
from jmdst.pipeline.models import FELNetLocalizer, YoloDetector, load_felnet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo", required=True, help="Trained YOLOv11 weights (.pt).")
    parser.add_argument("--felnet", required=True, help="Trained FELNet checkpoint (.pt).")
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--output-root", default="outputs/jmdst_results")
    parser.add_argument("--datasets", nargs="+", default=["visdrone", "uavdt"])
    parser.add_argument("--split", default="val")
    add_tracker_arguments(parser)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames per sequence (debugging).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from ultralytics import YOLO

    det_conf = detector_conf(args)
    detector = YoloDetector(YOLO(args.yolo), conf=det_conf, iou=args.iou, imgsz=args.imgsz, device=device)
    localizer = FELNetLocalizer(load_felnet(args.felnet, device=device), device=device)
    print(f"Loaded YOLO + FELNet on {device}. tau={args.tau}"
          + (f", BYTE [{det_conf}, {args.conf}) -> IoU<={args.byte_iou}" if args.byte else ""))

    output_root = Path(args.output_root)
    start = time.perf_counter()
    total_frames = 0

    for dataset in args.datasets:
        sequence_dirs = iter_sequence_dirs(args.unified_root, dataset=dataset, split=args.split)
        # Per-dataset timing (paper reports FPS per dataset). Timing covers only
        # pipeline inference, excluding image decode, so it reflects tracker speed.
        dataset_frames = 0
        dataset_seconds = 0.0
        for sequence_dir in sequence_dirs:
            info, records = read_sequence(sequence_dir)
            image_size = (info.image_width, info.image_height)
            # Fresh pipeline per sequence (reuses the loaded models).
            jmdst = tracker_from_args(args, detector, localizer)

            lines: list[str] = []
            frames = records if args.max_frames is None else records[: args.max_frames]
            for record in frames:
                image = Image.open(resolve_image_path(sequence_dir, record.image_path)).convert("RGB")
                t0 = time.perf_counter()
                outputs = jmdst.process_frame(image, image_size)
                dataset_seconds += time.perf_counter() - t0
                lines.extend(mot_lines(record.frame_id, outputs))
                total_frames += 1
                dataset_frames += 1

            out_path = output_root / dataset / args.split / f"{sequence_dir.name}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            fps = dataset_frames / dataset_seconds if dataset_seconds > 0 else 0.0
            print(
                f"{dataset}/{args.split}/{sequence_dir.name}: {len(frames)} frames, "
                f"{len(lines)} outputs -> {out_path}  ({fps:.1f} FPS avg)"
            )

        # Record per-dataset timing beside its results so eval can report FPS.
        dataset_fps = dataset_frames / dataset_seconds if dataset_seconds > 0 else 0.0
        timing_path = output_root / dataset / args.split / "timing.json"
        timing_path.write_text(
            json.dumps({"frames": dataset_frames, "seconds": dataset_seconds, "fps": dataset_fps, "tau": args.tau}, indent=2),
            encoding="utf-8",
        )

    print(f"\nDone: {total_frames} frames in {time.perf_counter()-start:.0f}s (wall).")


if __name__ == "__main__":
    main()
