"""Export a self-contained raw video clip for the live demo.

Why this exists: unified sequence directories hold only annotations, and the
image paths inside them are **absolute** (``C:/.../Datasets/UAVDT/...``). So
``demo.py --source data/unified/...`` works on the machine that built the
dataset and nowhere else — not on a borrowed presentation laptop, not from a
fresh clone, even if the raw datasets were copied to a different location.

Exporting a clip bakes the pixels into an mp4, after which the demo package is
just:

    scripts/demo.py + two checkpoints (~57 MB) + the clip

and carries no dependency on the datasets at all.

    python scripts/export_demo_clip.py --source data/unified/uavdt/test/M0101 \\
        --output demo_package/clips/uavdt_M0101.mp4 --max-frames 400

Note these are RAW clips, with no boxes drawn — the point is to run inference
on them live. Use ``demo.py --output`` if you want a pre-rendered annotated
video for a fallback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from demo import open_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, help="Unified sequence dir, image dir, or video.")
    parser.add_argument("--output", required=True, help="Destination .mp4.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Downscale if wider than this. VisDrone frames reach 2720px, which is "
             "slower to run and larger to carry than a demo needs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frames, src_fps, total = open_source(Path(args.source), args.max_frames, args.fps)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    count = 0
    try:
        for frame_bgr in frames:
            if args.max_width and frame_bgr.shape[1] > args.max_width:
                scale = args.max_width / frame_bgr.shape[1]
                frame_bgr = cv2.resize(
                    frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            if writer is None:
                h, w = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
                )
                if not writer.isOpened():
                    raise SystemExit(f"Could not open video writer for {out_path}")
                print(f"writing {w}x{h} @ {src_fps:.0f} fps -> {out_path}")
            writer.write(frame_bgr)
            count += 1
            if count % 100 == 0:
                print(f"  {count} frames...", flush=True)
    finally:
        if writer is not None:
            writer.release()

    size_mb = out_path.stat().st_size / 1024 ** 2 if out_path.exists() else 0.0
    print(f"done: {count} frames, {size_mb:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()
