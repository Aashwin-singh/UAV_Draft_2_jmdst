"""Telemetry report for an edge/ground simulation run (snapshot 4.11).

Joins the three logs a run leaves behind, all on one clock:

* the edge's ``exchanges.csv`` -- per detection frame: send time, ground
  arrival, ground compute, reply arrival, bytes, packets, ok/lost/late;
* the link emulator's ``link.csv`` -- per packet: queue delay and whether and
  why it was dropped, matched to its detection frame through the protocol
  header the link log records (like a packet capture);
* the edge's ``timing.json`` / ``frames.csv`` and, if present, ``eval.json``.

Prints one row per detection frame -- send time, arrival time, queue delay,
bytes, loss flag and why -- followed by a run summary, and writes the joined
table to ``telemetry.csv`` beside the edge's logs. Times are seconds from the
first send.

Usage (one line in cmd.exe):
    python scripts/sim_telemetry.py --run outputs/sim/demo
    python scripts/sim_telemetry.py --run outputs/sim/demo --only lost --rows 15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from jmdst.net.protocol import DETECT_REQUEST, DETECT_RESPONSE, FRAGMENT_HEADER

JOINED_FIELDS = [
    "sequence", "frame_id", "msg_id", "send_s", "arrival_s", "uplink_ms", "queue_delay_ms", "bytes",
    "packets", "packets_dropped", "lost", "reason", "ground_compute_ms", "rtt_ms",
]


def _f(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def load_link(path: Path) -> dict[tuple[str, int, int], list[dict]]:
    """Packets per (direction, message type, msg_id), from the link's log."""

    packets: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            head = bytes.fromhex(row.get("head") or "")
            if len(head) < FRAGMENT_HEADER.size:
                continue
            magic, _version, msg_type, msg_id, _index, _count = FRAGMENT_HEADER.unpack_from(head)
            if magic == b"JM":
                packets[(row["direction"], msg_type, msg_id)].append(row)
    return packets


def join(exchanges: list[dict], link, deadline_ms: float) -> list[dict]:
    rows = []
    for ex in exchanges:
        msg_id = int(ex["msg_id"])
        req = link.get(("e2g", DETECT_REQUEST, msg_id), []) if link is not None else []
        resp = link.get(("g2e", DETECT_RESPONSE, msg_id), []) if link is not None else []
        req_dropped = sum(r["dropped"] == "1" for r in req)
        resp_dropped = sum(r["dropped"] == "1" for r in resp)
        queue = max((float(r["queue_delay_ms"]) for r in req), default=None) if req else None
        if ex["outcome"] == "ok":
            reason = "-"
        elif req_dropped:
            reason = f"request: {req_dropped}/{len(req)} pkts lost"
        elif resp_dropped:
            reason = "reply lost on link"
        elif ex["outcome"] == "late":
            reason = f"late (> {deadline_ms:.0f} ms)"
        else:
            reason = "no reply"
        rows.append({
            "sequence": ex["sequence"], "frame_id": int(ex["frame_id"]), "msg_id": msg_id,
            "send_s": float(ex["send_s"]), "arrival_s": _f(ex["ground_arrival_s"]),
            "uplink_ms": _f(ex["uplink_ms"]), "queue_delay_ms": queue, "bytes": int(ex["request_bytes"]),
            "packets": int(ex["request_packets"]), "packets_dropped": req_dropped + resp_dropped,
            "lost": 0 if ex["outcome"] == "ok" else 1, "reason": reason,
            "ground_compute_ms": _f(ex["ground_compute_ms"]), "rtt_ms": _f(ex["rtt_ms"]),
        })
    return rows


def _cell(value, fmt: str, width: int) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}{fmt}}"


def print_table(rows: list[dict], t0: float, limit: int) -> None:
    print(f"{'sequence':<20} {'frame':>5} {'send s':>8} {'arrive s':>8} {'uplink ms':>9} {'queue ms':>8} "
          f"{'bytes':>7} {'pkts':>4} {'lost':>4}  {'reason':<26} {'RTT ms':>7}")
    print("-" * 118)
    for r in rows[:limit]:
        arrive = None if r["arrival_s"] is None else r["arrival_s"] - t0
        print(f"{r['sequence'][:20]:<20} {r['frame_id']:>5} {r['send_s'] - t0:>8.3f} {_cell(arrive, '.3f', 8)} "
              f"{_cell(r['uplink_ms'], '.1f', 9)} {_cell(r['queue_delay_ms'], '.1f', 8)} {r['bytes']:>7} "
              f"{r['packets']:>4} {r['lost']:>4}  {r['reason']:<26} {_cell(r['rtt_ms'], '.1f', 7)}")
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows in telemetry.csv")


def _pcts(values: list[float]) -> str:
    return f"p50 {np.percentile(values, 50):.1f} p95 {np.percentile(values, 95):.1f} ms" if values else "n/a"


def print_summary(run: Path, split_dir: Path, rows: list[dict], link_rows, link_cfg: dict | None) -> None:
    timing = json.loads((split_dir / "timing.json").read_text(encoding="utf-8"))
    print(f"\nRun summary -- {split_dir.relative_to(PROJECT_ROOT) if split_dir.is_relative_to(PROJECT_ROOT) else split_dir}")
    if link_cfg:
        print(f"  link      edge->ground: {link_cfg.get('e2g') or '(perfect link)'}")
        print(f"            ground->edge: {link_cfg.get('g2e') or '(perfect link)'}")
    else:
        print("  link      none (edge talked to the ground directly)")
    mode = (f"real time, camera {timing['camera_fps']:g} fps" if timing.get("mode") == "realtime" else "offline")
    print(f"  edge      {timing.get('edge_hardware')}; FELNet on {timing.get('edge_device')}; {mode}; "
          f"used {timing.get('edge_cpu_cores_used', float('nan')):.2f} cores; codec {timing.get('codec')}, "
          f"deadline {timing.get('deadline_ms'):.0f} ms")
    frames, processed = timing["frames"], timing.get("processed", timing["frames"])
    sent = len(rows)
    fallback = sum(r["lost"] for r in rows)
    print(f"  frames    {frames} from camera, {processed} processed ({frames - processed} skipped, "
          f"{100 * (frames - processed) / frames:.1f}%)")
    print(f"            {sent} sent to ground, {processed - sent} tracked locally; "
          f"{fallback} of the sent fell back to local tracking")
    outcomes = Counter("ok" if not r["lost"] else ("late" if r["reason"].startswith("late") else "lost") for r in rows)
    reasons = Counter(r["reason"].split(":")[0] for r in rows if r["lost"])
    print(f"  replies   ok {outcomes['ok']} ({100 * outcomes['ok'] / max(sent, 1):.1f}%) | "
          f"lost {outcomes['lost']} | late {outcomes['late']}"
          + (f"  [{', '.join(f'{k}: {v}' for k, v in reasons.items())}]" if reasons else ""))
    print(f"  latency   uplink {_pcts([r['uplink_ms'] for r in rows if r['uplink_ms'] is not None])} | "
          f"queue {_pcts([r['queue_delay_ms'] for r in rows if r['queue_delay_ms'] is not None])}")
    print(f"            RTT {_pcts([r['rtt_ms'] for r in rows if r['rtt_ms'] is not None])} | "
          f"ground compute {_pcts([r['ground_compute_ms'] for r in rows if r['ground_compute_ms'] is not None])}")
    if timing.get("latency_ms"):
        print(f"            frame latency, capture -> result: p50 {timing['latency_ms']['p50']:.1f} "
              f"p95 {timing['latency_ms']['p95']:.1f} ms")
    if len(rows) > 1:
        span = rows[-1]["send_s"] - rows[0]["send_s"]
        total = sum(r["bytes"] for r in rows)
        print(f"  traffic   edge->ground {total / 1e6:.2f} MB in {sum(r['packets'] for r in rows)} packets, "
              f"{total / sent / 1e3:.0f} KB per frame, {total * 8 / span / 1e6:.2f} Mbit/s average over {span:.1f} s")
    if link_rows is not None:
        e2g = [r for (d, _t, _m), pk in link_rows.items() if d == "e2g" for r in pk]
        drops = Counter(r["drop_reason"] for r in e2g if r["dropped"] == "1")
        print(f"            link dropped {sum(drops.values())} of {len(e2g)} edge->ground packets "
              f"({100 * sum(drops.values()) / max(len(e2g), 1):.2f}%)"
              + (f" [{', '.join(f'{k} {v}' for k, v in drops.items())}]" if drops else ""))
    eval_path = split_dir / "eval.json"
    if eval_path.exists():
        o = json.loads(eval_path.read_text(encoding="utf-8"))["overall"]
        print(f"  accuracy  HOTA {o['HOTA'] * 100:.1f}  MOTA {o['MOTA'] * 100:.1f}  IDF1 {o['IDF1'] * 100:.1f}  "
              f"IDs {o['IDs']}")
    elif split_dir.parent.name == "clip":
        print("  accuracy  n/a -- a video clip has no ground truth")
    else:
        print("  accuracy  not evaluated yet: run scripts/evaluate.py with --json <split dir>/eval.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Edge --output-root (run_sim.py --output-root).")
    parser.add_argument("--link-log", default=None, help="Link telemetry CSV (default: <run>/link.csv).")
    parser.add_argument("--rows", type=int, default=20, help="Table rows to print.")
    parser.add_argument("--only", choices=["all", "lost", "ok"], default="all")
    args = parser.parse_args(argv)

    run = (PROJECT_ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    link_path = Path(args.link_log) if args.link_log else run / "link.csv"
    link_rows = load_link(link_path) if link_path.exists() else None
    link_cfg = None
    if link_path.with_suffix(".json").exists():
        link_cfg = json.loads(link_path.with_suffix(".json").read_text(encoding="utf-8"))
    elif (run / "run.json").exists():
        meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
        link_cfg = None if meta.get("no_link") else {"e2g": meta.get("e2g"), "g2e": meta.get("g2e")}

    split_dirs = sorted(p.parent for p in run.glob("*/*/exchanges.csv"))
    if not split_dirs:
        raise SystemExit(f"no exchanges.csv under {run}: is this an edge_node.py --output-root?")
    for split_dir in split_dirs:
        timing = json.loads((split_dir / "timing.json").read_text(encoding="utf-8"))
        with open(split_dir / "exchanges.csv", encoding="utf-8") as f:
            exchanges = list(csv.DictReader(f))
        rows = join(exchanges, link_rows, timing.get("deadline_ms", float("nan")))
        with open(split_dir / "telemetry.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=JOINED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        shown = rows if args.only == "all" else [r for r in rows if r["lost"] == (args.only == "lost")]
        t0 = rows[0]["send_s"] if rows else 0.0
        print(f"\nDetection-frame telemetry -- {split_dir.parent.name}/{split_dir.name} "
              f"({len(rows)} frames sent to the ground; times in s from the first send)\n")
        print_table(shown, t0, args.rows)
        print_summary(run, split_dir, rows, link_rows, link_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
