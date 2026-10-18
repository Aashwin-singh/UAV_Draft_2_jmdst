"""Ground node: load YOLO, listen for detection requests from the edge, serve them.

The ground station is where the GPU detector lives. It receives a detection
frame from the edge (the UAV), runs YOLO with the post-processing the edge
asked for (conf / NMS IoU / imgsz travel in the request, so the two nodes
cannot disagree), and replies with the boxes. Everything else -- the tracker,
FELNet, the tau schedule -- stays on the edge.

Usage (one line in cmd.exe):
    python scripts/ground_node.py --yolo outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt

Stop with Ctrl+C; a summary is printed on exit.
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

import torch
from PIL import Image

from jmdst.net import GroundServer, high_resolution_timer, interrupt_on_ctrl_break
from jmdst.pipeline.models import YoloDetector


def _addr(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host or "127.0.0.1", int(port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo", required=True, help="Trained YOLOv11 weights (.pt).")
    parser.add_argument("--listen", default="127.0.0.1:47001", help="Address to serve on.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--telemetry", default=None, help="Per-request CSV log path.")
    parser.add_argument("--print-every", type=int, default=30, help="Print one 'serving' line per N requests (0 = none).")
    parser.add_argument("--cpus", default=None, help="Pin to these logical CPUs, e.g. 0-15, away from the edge's.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    interrupt_on_ctrl_break()
    if args.cpus:
        import psutil

        from jmdst.sim import parse_cpu_list

        psutil.Process().cpu_affinity(list(parse_cpu_list(args.cpus)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from ultralytics import YOLO

    print(f"[ground] loading YOLO {args.yolo} ...", flush=True)
    yolo = YoloDetector(YOLO(args.yolo), device=device)
    # Warm up every network input shape we will see, or the first request of
    # each shape pays CUDA kernel selection (~150 ms measured) and misses a
    # 100 ms deadline -- a "lost" frame that is really a cold start. Letterbox
    # to imgsz 640 maps every val/test frame size to one of two shapes:
    # 1024x540 -> 640x352 (UAVDT), 16:9 VisDrone sizes -> 640x384.
    t0 = time.perf_counter()
    for size in ((1024, 540), (1920, 1080)):
        for _ in range(2):
            yolo(Image.new("RGB", size))
    print(f"[ground] loaded on {device} (warm-up {(time.perf_counter() - t0) * 1e3:.0f} ms)", flush=True)

    def detect(image, conf, iou, imgsz):
        yolo.conf, yolo.iou, yolo.imgsz = conf, iou, imgsz
        return yolo(image)

    def on_event(kind: str, data: dict) -> None:
        if kind == "served" and args.print_every and data["msg_id"] % args.print_every == 0:
            print(
                f"[ground] serving frame {data['frame_id']} (msg {data['msg_id']}): "
                f"{data['request_bytes'] / 1e3:.1f} KB in {data['packets']} packets -> "
                f"{data['detections']} detections, YOLO {data['detect_ms']:.1f} ms",
                flush=True,
            )
        elif kind == "incomplete":
            print(f"[ground] request msg {data['msg_id']} never completed: "
                  f"{data['received']}/{data['expected']} packets arrived (lost on the link)", flush=True)

    log_file = None
    if args.telemetry:
        Path(args.telemetry).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.telemetry, "w", newline="", encoding="utf-8")
    server = GroundServer(
        _addr(args.listen), detect,
        info={"model": Path(args.yolo).parts[-3] if len(Path(args.yolo).parts) >= 3 else args.yolo, "device": device},
        telemetry=log_file, on_event=on_event,
    )
    print(f"[ground] listening on {args.listen}", flush=True)
    try:
        with high_resolution_timer():
            server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        s = server.stats
        mean = f"{statistics.fmean(s.detect_ms):.1f} ms" if s.detect_ms else "n/a"
        print(f"[ground] stopped. served {s.served} detection requests, {s.incomplete} arrived incomplete, "
              f"mean YOLO {mean}", flush=True)
        server.close()
        if log_file is not None:
            log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
