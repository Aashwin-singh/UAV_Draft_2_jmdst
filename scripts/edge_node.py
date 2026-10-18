"""Edge node: run JMDST on the UAV side, offloading only detection frames.

The edge runs the whole tracker -- camera-motion compensation, Kalman,
FELNet, association -- and owns the tau schedule. On a detection frame it
sends the frame to the ground node and waits up to ``--deadline-ms`` for the
boxes; if they do not come, that frame runs the tracking branch instead
(the paper's missed-detection mechanism). Tracking frames never leave the
aircraft. So of every tau frames, one is *sent* and tau-1 are *local*.

Two ways to run
---------------
* **Offline** (default): every frame is processed, however long it takes.
  Accuracy is unaffected by speed; FPS is the result.
* **Real time** (``--realtime``): a camera process publishes frames at
  ``--camera-fps`` whether or not the edge is ready, and the edge always
  takes the newest one -- frames it is too busy for are skipped (tracks coast
  on the Kalman model). Scoring follows the streaming-perception protocol
  (Li et al., ECCV 2020): camera frame j is scored with the latest result the
  edge had *finished* by j's capture time. A slow edge therefore costs
  accuracy, as it would for an operator watching the ground display.

Edge hardware (``--edge-cpu-cores``, ``--edge-cpus``, ``--edge-memory-mb``,
``--edge-threads``, ``--device cpu``) confines this process to a smaller
computer -- see jmdst/sim/hardware.py. The camera is not charged to it.

Output layout matches scripts/run_jmdst.py -- MOT files and timing.json under
<output-root>/<dataset>/<split>/ -- so scripts/evaluate.py scores it unchanged.
Per dataset it also writes ``exchanges.csv`` (one row per detection frame:
send / ground-arrival / ground-compute / reply times on one clock, bytes,
packets, ok/lost/late) and ``frames.csv`` (one row per processed frame:
capture / start / finish time, latency, frames skipped before it, branch,
compute vs network-wait time).

Usage (one line in cmd.exe; start the ground node and link first):
    python scripts/edge_node.py --felnet outputs/felnet_runs/full_run/best.pt --datasets uavdt --split val --conf 0.55 --no-appearance --nsa --cmc --byte --byte-low-conf 0.4 --realtime --device cpu --edge-cpus 16-19

``--ground`` defaults to the link emulator's edge port (127.0.0.1:47000);
point it at the ground node itself (127.0.0.1:47001) to bypass the link.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from jmdst.data.io import iter_sequence_dirs, read_sequence, resolve_image_path
from jmdst.net import EXCHANGE_FIELDS, RemoteDetector, exchange_row, high_resolution_timer
from jmdst.net.protocol import CODECS, DEFAULT_MTU
from jmdst.pipeline.cli import add_tracker_arguments, detector_conf, mot_lines, tracker_from_args
from jmdst.pipeline.models import FELNetLocalizer, load_felnet
from jmdst.sim import CameraFeed, CpuMeter, EdgeHardware, apply_edge_hardware, latest_finished, parse_cpu_list

FRAME_FIELDS = [
    "sequence", "camera_index", "frame_id", "capture_s", "start_s", "finish_s", "latency_ms",
    "skipped_before", "branch", "compute_ms", "network_wait_ms",
]


def _addr(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host or "127.0.0.1", int(port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--felnet", required=True, help="Trained FELNet checkpoint (.pt).")
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--output-root", default="outputs/sim/edge")
    parser.add_argument("--datasets", nargs="+", default=["visdrone", "uavdt"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--sequences", nargs="+", default=None, help="Only these sequence names.")
    add_tracker_arguments(parser)
    parser.add_argument("--device", default=None, help="cuda | cpu (a CPU-only edge).")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames per sequence (debugging).")
    net = parser.add_argument_group("network")
    net.add_argument("--ground", default="127.0.0.1:47000", help="Where to send detection frames (link emulator or ground).")
    net.add_argument("--codec", default="jpeg", choices=sorted(CODECS), help="Frame encoding; raw/png are lossless.")
    net.add_argument("--jpeg-quality", type=int, default=90)
    net.add_argument("--deadline-ms", type=float, default=100.0, help="Give up on a detection reply after this long.")
    net.add_argument("--mtu", type=int, default=DEFAULT_MTU, help="Max datagram bytes, protocol header included.")
    net.add_argument("--handshake-timeout", type=float, default=60.0, help="Seconds to wait for the ground at start-up.")
    hw = parser.add_argument_group("edge hardware")
    hw.add_argument("--edge-cpu-cores", type=float, default=None, help="Hard CPU cap, in cores' worth (e.g. 1.5).")
    hw.add_argument("--edge-cpus", default=None, help="Logical CPUs to run on, e.g. 16-19 (E-cores on an i9-14900HX).")
    hw.add_argument("--edge-memory-mb", type=int, default=None, help="Memory (commit) limit.")
    hw.add_argument("--edge-threads", type=int, default=None, help="torch/OpenCV threads (default: fits the limits).")
    rt = parser.add_argument_group("real time")
    rt.add_argument("--realtime", action="store_true", help="Camera on a fixed clock; skip frames the edge cannot keep up with.")
    rt.add_argument("--camera-fps", type=float, default=30.0,
                    help="UAVDT is published as 30 fps; VisDrone states no rate, 30 is assumed.")
    return parser


def _pct(values: list[float], q: float) -> str:
    return f"{np.percentile(values, q):.1f}" if values else "n/a"


def _frame_row(seq: str, k: int, frame_id: int, capture, start, finish, skipped, branch, wait_ms) -> list:
    return [
        seq, k, frame_id, "" if capture is None else f"{capture:.6f}", f"{start:.6f}", f"{finish:.6f}",
        "" if capture is None else f"{(finish - capture) * 1e3:.3f}", skipped, branch,
        f"{(finish - start) * 1e3 - wait_ms:.3f}", f"{wait_ms:.3f}",
    ]


def run_offline(jmdst, remote, sequence_dir, records, image_size) -> tuple[list[str], list[list], dict]:
    lines, rows, busy = [], [], 0.0
    for k, record in enumerate(records):
        image = Image.open(resolve_image_path(sequence_dir, record.image_path)).convert("RGB")
        remote.context = {"sequence": sequence_dir.name, "frame_id": record.frame_id}
        remote.last_wait_ms = 0.0
        start = time.perf_counter()
        outputs = jmdst.process_frame(image, image_size)
        finish = time.perf_counter()
        busy += finish - start
        lines.extend(mot_lines(record.frame_id, outputs))
        rows.append(_frame_row(sequence_dir.name, k, record.frame_id, None, start, finish, 0,
                               jmdst.last_branch, remote.last_wait_ms if jmdst.last_branch != "track" else 0.0))
    return lines, rows, {"processed": len(records), "busy_s": busy, "fps": len(records) / busy if busy else 0.0}


def run_realtime(jmdst, remote, sequence_dir, records, image_size, fps) -> tuple[list[str], list[list], dict]:
    paths = [resolve_image_path(sequence_dir, r.image_path) for r in records]
    feed = CameraFeed(paths, image_size, fps)
    results: list[tuple[float, list]] = []  # (finish time, outputs), in finish order
    rows: list[list] = []
    try:
        t0 = feed.start()
        last = -1
        while (taken := feed.take_newest(last)) is not None:
            k, image = taken
            start = time.perf_counter()
            for _ in range(k - last - 1):
                jmdst.skip_frame()
            remote.context = {"sequence": sequence_dir.name, "frame_id": records[k].frame_id}
            remote.last_wait_ms = 0.0
            outputs = jmdst.process_frame(image, image_size)
            finish = time.perf_counter()
            results.append((finish, outputs))
            rows.append(_frame_row(sequence_dir.name, k, records[k].frame_id, feed.capture_time(k), start, finish,
                                   k - last - 1, jmdst.last_branch,
                                   remote.last_wait_ms if jmdst.last_branch != "track" else 0.0))
            last = k
        camera_late = feed.max_lateness_ms
    finally:
        feed.close()

    # Streaming scoring: camera frame j shows the newest result finished by its
    # capture time. Before the first result exists, nothing is shown.
    shown = latest_finished([f for f, _ in results], [t0 + j / fps for j in range(len(records))])
    lines = []
    for record, i in zip(records, shown):
        if i >= 0:
            lines.extend(mot_lines(record.frame_id, results[i][1]))
    duration = len(records) / fps
    return lines, rows, {
        "processed": len(results), "skipped": len(records) - len(results),
        "fps": len(results) / duration, "camera_max_lateness_ms": camera_late,
        "latency_ms": [float(r[6]) for r in rows],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hardware = EdgeHardware(
        cpu_cores=args.edge_cpu_cores, cpus=parse_cpu_list(args.edge_cpus),
        memory_mb=args.edge_memory_mb, threads=args.edge_threads,
    )
    apply_edge_hardware(hardware)  # before any model is loaded, so loading is constrained too
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    localizer = FELNetLocalizer(load_felnet(args.felnet, device=device), device=device)
    print(f"[edge] hardware: {hardware.describe()}; FELNet on {device}", flush=True)
    print(f"[edge] tau={args.tau}: 1 frame in {args.tau} is sent to the ground, the rest are tracked locally. "
          + (f"Real time: camera at {args.camera_fps:g} fps, newest frame wins." if args.realtime
             else "Offline: every frame is processed."), flush=True)
    remote = RemoteDetector(
        _addr(args.ground), conf=detector_conf(args), iou=args.iou, imgsz=args.imgsz,
        codec=args.codec, jpeg_quality=args.jpeg_quality, deadline_s=args.deadline_ms / 1e3, mtu=args.mtu,
    )
    print(f"[edge] contacting ground via {args.ground} ...", flush=True)
    info = remote.handshake(timeout_s=args.handshake_timeout)
    print(f"[edge] ground ready: {info}. codec={args.codec}"
          + (f" q{args.jpeg_quality}" if args.codec == "jpeg" else "")
          + f", deadline {args.deadline_ms:.0f} ms", flush=True)

    output_root = Path(args.output_root)
    start = time.perf_counter()
    total_frames = 0
    meter = CpuMeter()
    with high_resolution_timer():
        for dataset in args.datasets:
            sequence_dirs = [
                d for d in iter_sequence_dirs(args.unified_root, dataset=dataset, split=args.split)
                if args.sequences is None or d.name in args.sequences
            ]
            split_dir = output_root / dataset / args.split
            split_dir.mkdir(parents=True, exist_ok=True)
            first_exchange = len(remote.exchanges)
            frame_rows: list[list] = []
            per_seq: dict[str, dict] = {}
            meter.reset()
            for sequence_dir in sequence_dirs:
                info_seq, records = read_sequence(sequence_dir)
                records = records if args.max_frames is None else records[: args.max_frames]
                image_size = (info_seq.image_width, info_seq.image_height)
                jmdst = tracker_from_args(args, remote, localizer)
                seq_first = len(remote.exchanges)

                if args.realtime:
                    lines, rows, stats = run_realtime(jmdst, remote, sequence_dir, records, image_size, args.camera_fps)
                else:
                    lines, rows, stats = run_offline(jmdst, remote, sequence_dir, records, image_size)
                remote.drain(wait_s=0.2)  # classify any last reply that arrived after its deadline
                frame_rows.extend(rows)

                (split_dir / f"{sequence_dir.name}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

                exchanges = remote.exchanges[seq_first:]
                outcomes = {k: sum(e.outcome == k for e in exchanges) for k in ("ok", "lost", "late")}
                rtts = [e.rtt_ms for e in exchanges if e.outcome == "ok"]
                stats.update({"frames": len(records), "outcomes": outcomes, "detection_frames": len(exchanges)})
                per_seq[sequence_dir.name] = {k: v for k, v in stats.items() if k != "latency_ms"}
                pace = (f"processed {stats['processed']}/{len(records)} ({stats['skipped']} skipped), "
                        f"latency p50 {_pct(stats['latency_ms'], 50)} p95 {_pct(stats['latency_ms'], 95)} ms"
                        if args.realtime else f"{stats['fps']:.1f} FPS")
                print(
                    f"[edge] {dataset}/{args.split}/{sequence_dir.name}: {len(records)} frames | "
                    f"sent {len(exchanges)} to ground (ok {outcomes['ok']}, lost {outcomes['lost']}, "
                    f"late {outcomes['late']}) | local {stats['processed'] - len(exchanges)} "
                    f"+ {jmdst.unavailable_detection_frames} fallback | RTT p50 {_pct(rtts, 50)} ms | {pace}",
                    flush=True,
                )
                total_frames += len(records)

            cores_used = meter.cores_used()
            exchanges = remote.exchanges[first_exchange:]
            with open(split_dir / "exchanges.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(EXCHANGE_FIELDS)
                writer.writerows(exchange_row(e) for e in exchanges)
            with open(split_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(FRAME_FIELDS)
                writer.writerows(frame_rows)

            frames = sum(s["frames"] for s in per_seq.values())
            processed = sum(s["processed"] for s in per_seq.values())
            busy = [float(r[5]) - float(r[4]) for r in frame_rows]
            rtts = [e.rtt_ms for e in exchanges if e.outcome == "ok"]
            timing = {
                # Same keys as run_jmdst.py's timing.json, so evaluate.py reads it unchanged.
                # Offline: processing FPS. Real time: processed frames per second of camera time.
                "frames": frames, "seconds": sum(busy) if not args.realtime else frames / args.camera_fps,
                "fps": (processed / sum(busy) if busy and not args.realtime else processed / (frames / args.camera_fps)),
                "tau": args.tau,
                "mode": "realtime" if args.realtime else "offline",
                "camera_fps": args.camera_fps if args.realtime else None,
                "processed": processed, "skipped": frames - processed,
                "edge_hardware": hardware.describe(), "edge_device": device,
                "edge_cpu_cores_used": cores_used,
                "detection_frames": len(exchanges),
                "outcomes": {k: sum(e.outcome == k for e in exchanges) for k in ("ok", "lost", "late")},
                "rtt_ms": {"p50": float(np.percentile(rtts, 50)), "p95": float(np.percentile(rtts, 95))} if rtts else None,
                "codec": args.codec, "deadline_ms": args.deadline_ms,
                "sequences": per_seq,
            }
            if args.realtime:
                lat = [float(r[6]) for r in frame_rows]
                timing["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
                timing["camera_max_lateness_ms"] = max(s["camera_max_lateness_ms"] for s in per_seq.values())
            (split_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
            print(f"[edge] {dataset}: edge used {cores_used:.2f} cores on average"
                  + (f"; processed {processed}/{frames} frames ({100 * (frames - processed) / frames:.1f}% skipped)"
                     if args.realtime else ""), flush=True)

    remote.close()
    print(f"\n[edge] done: {total_frames} frames in {time.perf_counter() - start:.0f}s (wall).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
