"""Run the network link emulator between the edge and ground nodes.

The edge node sends to ``--edge-listen``; the emulator forwards to the ground
node at ``--ground`` through the edge->ground link, and carries the ground's
replies back through the ground->edge link. Each direction takes a netem
option string (see ``jmdst/net/link.py``), so a Linux cross-check can apply
the identical strings with ``tc qdisc add dev <if> root netem <spec>``.

Usage (one line in cmd.exe):
    python scripts/link_relay.py --e2g "delay 40ms 10ms rate 8mbit loss 1%" --g2e "delay 40ms" --telemetry outputs/sim/link.csv

Stop with Ctrl+C; a per-direction summary is printed on exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.net import LinkRelay, interrupt_on_ctrl_break, parse_netem


def _addr(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host or "127.0.0.1", int(port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edge-listen", default="127.0.0.1:47000", help="Address the edge node sends to.")
    parser.add_argument("--ground", default="127.0.0.1:47001", help="Address of the ground node.")
    parser.add_argument("--e2g", default="", help='Edge->ground netem spec, e.g. "delay 40ms 10ms rate 8mbit loss 1%%".')
    parser.add_argument("--g2e", default="", help="Ground->edge netem spec.")
    parser.add_argument("--telemetry", default=None, help="Per-packet CSV log path.")
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds.")
    parser.add_argument("--stats-every", type=float, default=5.0, help="Seconds between status lines.")
    parser.add_argument("--cpus", default=None, help="Pin to these logical CPUs, e.g. 0-15, away from the edge's.")
    return parser


def _status(relay: LinkRelay) -> str:
    stats = relay.stats()
    parts = []
    for direction in ("e2g", "g2e"):
        s = stats[direction]
        lost = s["dropped_queue"] + s["dropped_channel"] + s["dropped_host"]
        parts.append(f"{direction}: {s['offered']} in / {s['delivered']} out / {lost} lost / {s['in_flight']} in flight")
    if "release_error_ms" in stats:
        parts.append(f"timing error p95 {stats['release_error_ms']['p95']:.2f} ms")
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    interrupt_on_ctrl_break()
    if args.cpus:
        import psutil

        from jmdst.sim import parse_cpu_list

        psutil.Process().cpu_affinity(list(parse_cpu_list(args.cpus)))
    e2g, g2e = parse_netem(args.e2g), parse_netem(args.g2e)

    log_file = None
    if args.telemetry:
        Path(args.telemetry).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.telemetry, "w", newline="", encoding="utf-8")
        # The link's settings travel with its log, so a report can state them.
        Path(args.telemetry).with_suffix(".json").write_text(
            json.dumps({"e2g": e2g.to_netem(), "g2e": g2e.to_netem()}, indent=2), encoding="utf-8")

    relay = LinkRelay(_addr(args.edge_listen), _addr(args.ground), e2g, g2e, telemetry=log_file)
    print(f"[link] edge->ground: {e2g.to_netem()}")
    print(f"[link] ground->edge: {g2e.to_netem()}")
    print(f"[link] listening for edge on {args.edge_listen}, forwarding to ground at {args.ground}", flush=True)
    elapsed = 0.0
    try:
        while args.duration is None or elapsed < args.duration:
            step = args.stats_every if args.duration is None else min(args.stats_every, args.duration - elapsed)
            relay.run(duration=step)
            elapsed += step
            print(f"[link] {_status(relay)}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("[link] stopped. summary:")
        print(json.dumps(relay.stats(), indent=2))
        relay.close()
        if log_file is not None:
            log_file.close()
            print(f"[link] per-packet telemetry: {args.telemetry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
