"""Verify a real-time edge run by deterministic replay.

A real-time run's output depends on wall-clock timing, so it cannot be
compared with a reference directly. But everything *except* timing is
deterministic: the ground's detections depend only on the frame's encoded
bytes. So replay offline -- process exactly the camera frames the run
processed, ``skip_frame()`` for the ones it skipped, run the ground's
encode -> decode -> YOLO locally -- then apply the run's recorded finish times
through ``latest_finished``. If the camera, frame skipping and streaming
scoring are correct, the MOT files must match byte for byte.

Reads the run's own settings from ``run.json`` (written by run_sim.py).
Valid for runs whose detection replies were all ``ok`` (a lost reply cannot
be replayed without the link's loss pattern); the script refuses others.

Usage (one line in cmd.exe):
    python scripts/verify_realtime_replay.py --run outputs/sim/rtval_gpu --dataset uavdt --split val
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from PIL import Image

from edge_node import build_parser as edge_parser
from jmdst.data.io import read_sequence, resolve_image_path
from jmdst.net.protocol import CODECS, decode_image, encode_image
from jmdst.pipeline.cli import detector_conf, mot_lines, tracker_from_args
from jmdst.pipeline.models import FELNetLocalizer, YoloDetector, load_felnet
from jmdst.sim import latest_finished


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run_sim.py --output-root of a --realtime run.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--unified-root", default="data/unified")
    cli = parser.parse_args(argv)

    run_root = PROJECT_ROOT / cli.run
    meta = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    args = edge_parser().parse_args(["--felnet", meta["felnet"], *meta["edge_args"]])
    if not args.realtime:
        raise SystemExit("not a --realtime run; compare offline runs with a plain diff instead")
    split_dir = run_root / cli.dataset / cli.split
    timing = json.loads((split_dir / "timing.json").read_text(encoding="utf-8"))
    if timing["outcomes"]["lost"] or timing["outcomes"]["late"]:
        raise SystemExit(f"run has lost/late detection replies {timing['outcomes']}; replay needs all ok")

    import torch
    from ultralytics import YOLO

    from jmdst.sim import EdgeHardware, parse_cpu_list

    # Each model runs where it ran in the deployment: YOLO on the ground's
    # device, FELNet on the edge's -- with the edge's thread count, since CPU
    # float reductions can round differently with a different number of threads.
    threads = EdgeHardware(cpu_cores=args.edge_cpu_cores, cpus=parse_cpu_list(args.edge_cpus),
                           threads=args.edge_threads).thread_count()
    if threads is not None:
        import cv2

        torch.set_num_threads(threads)
        cv2.setNumThreads(threads)
    cuda = "cuda" if torch.cuda.is_available() else "cpu"
    ground_device = meta.get("ground_device") or cuda
    edge_device = args.device or cuda
    yolo = YoloDetector(YOLO(meta["yolo"]), conf=detector_conf(args), iou=args.iou, imgsz=args.imgsz, device=ground_device)

    def detector(image):  # exactly what the ground does with the edge's request
        return yolo(decode_image(CODECS[args.codec], encode_image(image, args.codec, args.jpeg_quality)))

    localizer = FELNetLocalizer(load_felnet(meta["felnet"], device=edge_device), device=edge_device)
    rows = list(csv.DictReader(open(split_dir / "frames.csv", encoding="utf-8")))
    sequences = sorted({r["sequence"] for r in rows})
    identical = 0
    for seq in sequences:
        seq_rows = [r for r in rows if r["sequence"] == seq]
        seq_dir = PROJECT_ROOT / cli.unified_root / cli.dataset / cli.split / seq
        info, records = read_sequence(seq_dir)
        records = records if args.max_frames is None else records[: args.max_frames]
        jmdst = tracker_from_args(args, detector, localizer)
        results = []
        for r in seq_rows:
            for _ in range(int(r["skipped_before"])):
                jmdst.skip_frame()
            k = int(r["camera_index"])
            image = Image.open(resolve_image_path(seq_dir, records[k].image_path)).convert("RGB")
            results.append((float(r["finish_s"]), jmdst.process_frame(image, (info.image_width, info.image_height))))
        t0 = float(seq_rows[0]["capture_s"]) - int(seq_rows[0]["camera_index"]) / args.camera_fps
        shown = latest_finished([f for f, _ in results], [t0 + j / args.camera_fps for j in range(len(records))])
        lines = []
        for record, i in zip(records, shown):
            if i >= 0:
                lines.extend(mot_lines(record.frame_id, results[i][1]))
        replay = "\n".join(lines) + ("\n" if lines else "")
        ok = replay == (split_dir / f"{seq}.txt").read_text(encoding="utf-8")
        identical += ok
        print(f"{seq}: replay {'IDENTICAL to' if ok else 'DIFFERS from'} the real-time output ({len(lines)} lines)")
    print(f"{identical}/{len(sequences)} sequences identical")
    return 0 if identical == len(sequences) else 1


if __name__ == "__main__":
    raise SystemExit(main())
