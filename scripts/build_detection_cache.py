"""Build the modelled ground station's detection cache (see jmdst/net/detcache.py).

For every frame the edge could send, reproduce the ground's exact work --
encode the frame as the edge does, decode it, run YOLO with the edge's
post-processing -- and record the detections plus that frame's own decode +
detect time. Run it on an **idle machine**: those times become the ground's
compute model for every sweep. The script records the CPU load it started
under and refuses to time on a busy machine unless told to.

Usage (one line in cmd.exe; ~4 min for val, ~1 min for the three demo clips):
    python scripts/build_detection_cache.py --yolo outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt --datasets uavdt visdrone --split val --conf 0.55 --byte --byte-low-conf 0.4 --output outputs/sim/detcache/val_jpeg90.npz

Use the same tracker options as the runs it will serve: only the detector's
conf / IoU / imgsz matter here (BYTE lowers the detector's conf), and a
request asking for anything else is a cache miss, never a wrong answer.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
import torch
from PIL import Image

from jmdst.data.io import iter_sequence_dirs, read_sequence, resolve_image_path
from jmdst.net import DetectionCache
from jmdst.net.protocol import CODECS, decode_image, encode_image
from jmdst.pipeline.cli import add_tracker_arguments, detector_conf
from jmdst.pipeline.models import YoloDetector


def _dataset_frames(args):
    for dataset in args.datasets:
        for seq_dir in iter_sequence_dirs(args.unified_root, dataset=dataset, split=args.split):
            _info, records = read_sequence(seq_dir)
            records = records if args.max_frames is None else records[: args.max_frames]
            for r in records:
                # Same decode as the edge (offline) and the camera process (real time).
                yield f"{dataset}/{seq_dir.name}", Image.open(resolve_image_path(seq_dir, r.image_path)).convert("RGB")


def _clip_frames(args):
    import cv2

    for clip in args.sources:
        cap = cv2.VideoCapture(clip)
        n = 0
        while args.max_frames is None or n < args.max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            n += 1
            yield f"clip/{Path(clip).stem}", Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--split", default="val")
    parser.add_argument("--sources", nargs="*", default=[], help="Video clips, e.g. demo_package/clips/*.mp4.")
    add_tracker_arguments(parser)
    parser.add_argument("--codec", default="jpeg", choices=sorted(CODECS))
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-busy", action="store_true", help="Time even if the machine is not idle.")
    args = parser.parse_args(argv)
    if not args.datasets and not args.sources:
        parser.error("give --datasets and/or --sources")

    load = psutil.cpu_percent(interval=2.0)
    if load > 15 and not args.allow_busy:
        raise SystemExit(f"machine is {load:.0f}% busy: these timings become the ground's compute model, "
                         "so build the cache on an idle machine (or pass --allow-busy)")

    from ultralytics import YOLO

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    conf = detector_conf(args)
    yolo = YoloDetector(YOLO(args.yolo), conf=conf, iou=args.iou, imgsz=args.imgsz, device=device)
    for size in ((1024, 540), (1920, 1080)):  # warm both letterbox shapes, as ground_node.py does
        for _ in range(3):
            yolo(Image.new("RGB", size))

    cache = DetectionCache({
        "conf": conf, "iou": args.iou, "imgsz": args.imgsz, "codec": args.codec,
        "jpeg_quality": args.jpeg_quality, "yolo": args.yolo, "device": device,
        "cpu_busy_at_start_pct": load, "built": time.strftime("%Y-%m-%d %H:%M"),
    })
    codec_id = CODECS[args.codec]
    per_group: dict[str, list[float]] = {}
    start = time.perf_counter()
    frames = 0
    sources = []
    if args.datasets:
        sources.append(_dataset_frames(args))
    if args.sources:
        sources.append(_clip_frames(args))
    for frame_iter in sources:
        for group, image in frame_iter:
            data = encode_image(image, args.codec, args.jpeg_quality)
            t0 = time.perf_counter()
            detections = yolo(decode_image(codec_id, data))
            ms = (time.perf_counter() - t0) * 1e3
            cache.add(data, detections, ms)
            per_group.setdefault(group, []).append(ms)
            frames += 1
            if frames % 500 == 0:
                print(f"  {frames} frames cached ({time.perf_counter() - start:.0f}s)", flush=True)

    cache.save(args.output)
    print(f"\nCached {len(cache)} distinct frames ({frames} processed) -> {args.output}")
    print(f"Machine CPU busy at start: {load:.0f}%. Ground compute model (decode + YOLO), median per sequence:")
    for group, values in per_group.items():
        print(f"  {group:40s} {statistics.median(values):6.1f} ms  (p95 {sorted(values)[int(0.95 * (len(values) - 1))]:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
