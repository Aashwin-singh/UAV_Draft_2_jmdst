"""Run JMDST on a video (or image sequence) and render tracked boxes.

This is the demo/presentation entry point. Unlike ``run_jmdst.py``, which is a
batch evaluation script coupled to the unified dataset layout and emits
MOT-format text, this reads ordinary video and produces something watchable.

Kept out of ``jmdst/`` deliberately: rendering is presentation, not pipeline.
It drives the library through its normal API (``build_jmdst`` +
``process_frame``), so nothing here can affect measured results.

Sources (auto-detected):
  * a video file            -- demo.py --source clip.mp4
  * a directory of frames   -- demo.py --source frames/
  * a unified sequence dir  -- demo.py --source data/unified/uavdt/test/M0101

Usage (write an annotated video):
    python scripts/demo.py --source clip.mp4 --output out.mp4

Usage (live window, press q to quit):
    python scripts/demo.py --source data/unified/uavdt/test/M0101 --show

Defaults match the project's final configuration (see RESULTS_SUMMARY.md):
yolo11m low-LR detector, FELNet full_run, IoU-only association, NSA at
detection frames, ECC camera-motion compensation, conf 0.55, tau 3.

Note on the first frames: a tracklet is only output once confirmed, which
needs N = tau + 1 = 4 consecutive present frames. Expect no boxes for the
first few frames of any clip. That is the paper's rule, not a bug.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.pipeline.models import build_jmdst
from jmdst.tracking import CameraMotionCompensator

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_YOLO = "outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt"
DEFAULT_FELNET = "outputs/felnet_runs/full_run/best.pt"


# ---------------------------------------------------------------- frame sources


def _frames_from_video(path: Path, max_frames: int | None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    def generator():
        count = 0
        while max_frames is None or count < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            yield frame_bgr
            count += 1
        capture.release()

    return generator(), fps, (min(total, max_frames) if total and max_frames else total)


def _frames_from_paths(paths: list[Path], max_frames: int | None, fps: float):
    paths = paths if max_frames is None else paths[:max_frames]

    def generator():
        for p in paths:
            frame_bgr = cv2.imread(str(p))
            if frame_bgr is None:
                raise SystemExit(f"Could not read image: {p}")
            yield frame_bgr

    return generator(), fps, len(paths)


def open_source(source: Path, max_frames: int | None, fps: float):
    """Return (frame_generator_bgr, fps, total_frames_or_None)."""

    if source.is_file():
        if source.suffix.lower() in VIDEO_SUFFIXES:
            return _frames_from_video(source, max_frames)
        raise SystemExit(f"Unsupported file type: {source.suffix}")

    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")

    # A unified sequence directory: use its annotation order, which is
    # authoritative, rather than trusting filename sorting.
    if (source / "seqinfo.json").exists():
        from jmdst.data.io import read_sequence, resolve_image_path

        _info, records = read_sequence(source)
        paths = [Path(resolve_image_path(source, r.image_path)) for r in records]
        return _frames_from_paths(paths, max_frames, fps)

    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No images found in {source}")
    return _frames_from_paths(paths, max_frames, fps)


# ------------------------------------------------------------------- rendering


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR colour per track id."""

    hue = (int(track_id) * 47) % 180  # 47 is coprime with 180 -> spreads hues
    hsv = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def draw_outputs(frame_bgr: np.ndarray, outputs, thickness: int = 2) -> None:
    for out in outputs:
        x, y, w, h = (float(v) for v in out.bbox_xywh)
        x1, y1, x2, y2 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
        colour = track_color(out.track_id)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, thickness)

        label = f"{out.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Keep the label inside the frame when the box touches the top edge.
        top = max(y1, th + 4)
        cv2.rectangle(frame_bgr, (x1, top - th - 4), (x1 + tw + 4, top), colour, -1)
        cv2.putText(frame_bgr, label, (x1 + 2, top - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def warm_up(tracker, frame_bgr: np.ndarray, rounds: int = 3) -> float:
    """Pre-pay the CUDA cold start before the audience is watching.

    The first inference on a fresh CUDA context measures ~2.2 s against a ~19 ms
    steady state (116x). Left unwarmed, a live demo freezes on frame 0 for two
    seconds, which looks like a crash.

    Exercises the detector and localizer directly rather than
    ``process_frame``, so tracker state (tracks, frame_index, CMC's previous
    frame) is untouched and the demo still starts from a clean slate.
    """

    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    probe = [np.array([0.0, 0.0, 64.0, 64.0], dtype=np.float64)]
    started = time.perf_counter()
    for _ in range(rounds):
        tracker.detector(image)
        tracker.localizer(image, probe)
    return time.perf_counter() - started


def draw_hud(frame_bgr: np.ndarray, lines: list[str]) -> None:
    pad, line_h = 8, 22
    box_h = pad * 2 + line_h * len(lines)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (330, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)
    for i, text in enumerate(lines):
        cv2.putText(frame_bgr, text, (pad, pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, help="Video file, image directory, or unified sequence directory.")
    parser.add_argument("--output", default=None, help="Annotated .mp4 to write. Omit with --show for display only.")
    parser.add_argument("--show", action="store_true", help="Display a live window (press q to quit).")
    parser.add_argument("--yolo", default=DEFAULT_YOLO)
    parser.add_argument("--felnet", default=DEFAULT_FELNET)
    parser.add_argument("--tau", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0, help="Output frame rate for image-sequence sources.")
    parser.add_argument("--appearance", action="store_true", help="Use FELNet appearance matching (default is IoU-only, which measures better).")
    parser.add_argument("--no-nsa", action="store_true", help="Disable the NSA Kalman filter (on by default).")
    parser.add_argument("--no-cmc", action="store_true", help="Disable camera-motion compensation (on by default).")
    parser.add_argument("--no-hud", action="store_true", help="Hide the overlay panel.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help=(
            "Dummy inference rounds before frame 0, to pre-pay the CUDA cold start "
            "(~2.2s vs ~19ms steady state). Set 0 to disable."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.output and not args.show:
        raise SystemExit("Nothing to do: pass --output <file.mp4>, --show, or both.")

    for label, path in (("YOLO weights", args.yolo), ("FELNet checkpoint", args.felnet)):
        if not Path(path).exists():
            raise SystemExit(
                f"{label} not found: {path}\n"
                "Checkpoints live under outputs/ and are gitignored -- copy them in or retrain."
            )

    source = Path(args.source)
    frames, src_fps, total = open_source(source, args.max_frames, args.fps)

    tracker = build_jmdst(
        yolo_weights=args.yolo,
        felnet_checkpoint=args.felnet,
        tau=args.tau,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        use_appearance=args.appearance,
        use_nsa=not args.no_nsa,
        cmc=None if args.no_cmc else CameraMotionCompensator(),
    )

    writer = None
    inference_seconds = 0.0
    started = time.perf_counter()
    index = 0

    try:
        for frame_bgr in frames:
            height, width = frame_bgr.shape[:2]

            if index == 0 and args.warmup > 0:
                print(f"warming up ({args.warmup} rounds)...", flush=True)
                elapsed = warm_up(tracker, frame_bgr, args.warmup)
                print(f"  ready in {elapsed:.1f}s", flush=True)
                started = time.perf_counter()  # exclude warmup from the FPS report

            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            t0 = time.perf_counter()
            outputs = tracker.process_frame(image, (width, height))
            inference_seconds += time.perf_counter() - t0

            draw_outputs(frame_bgr, outputs)
            if not args.no_hud:
                inference_fps = (index + 1) / inference_seconds if inference_seconds else 0.0
                draw_hud(frame_bgr, [
                    f"frame {index}" + (f" / {total}" if total else ""),
                    # The dual-branch schedule is the paper's core idea, so make
                    # it visible: full detection only every tau-th frame.
                    "DETECT (YOLO+FELNet)" if index % args.tau == 0 else "TRACK (FELNet only)",
                    f"tracks: {len(outputs)}    {inference_fps:.1f} FPS (inference)",
                ])

            if args.output:
                if writer is None:
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise SystemExit(f"Could not open video writer for {args.output}")
                writer.write(frame_bgr)

            if args.show:
                cv2.imshow("JMDST", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            index += 1
            if index % 50 == 0:
                print(f"  {index} frames...", flush=True)
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - started
    print(f"\nProcessed {index} frames in {wall:.1f}s wall.")
    if inference_seconds:
        print(f"  inference: {index / inference_seconds:.1f} FPS "
              f"(end-to-end incl. decode/draw/encode: {index / wall:.1f} FPS)")
    if args.output:
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
