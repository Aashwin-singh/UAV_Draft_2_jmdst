"""Tests for the edge/ground telemetry join (scripts/sim_telemetry.py)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from jmdst.net.protocol import DETECT_REQUEST, DETECT_RESPONSE, FRAGMENT_HEADER, MAGIC, VERSION
from jmdst.net.relay import TELEMETRY_FIELDS
from scripts.sim_telemetry import join, load_link


def _exchange(msg_id: int, outcome: str, arrival: str = "", rtt: str = "") -> dict:
    return {
        "sequence": "S", "frame_id": str(msg_id * 3), "msg_id": str(msg_id), "outcome": outcome,
        "send_s": f"{msg_id}.0", "ground_arrival_s": arrival, "uplink_ms": "" if not arrival else "40.0",
        "request_bytes": "3000", "request_packets": "3", "ground_compute_ms": "", "rtt_ms": rtt,
    }


def _packet(direction: str, msg_type: int, msg_id: int, index: int, count: int, queue_ms: float,
            dropped: bool) -> list:
    head = FRAGMENT_HEADER.pack(MAGIC, VERSION, msg_type, msg_id, index, count).hex()
    row = dict.fromkeys(TELEMETRY_FIELDS, "")
    row.update(direction=direction, queue_delay_ms=f"{queue_ms:.3f}", dropped="1" if dropped else "0",
               drop_reason="channel" if dropped else "", head=head)
    return [row[f] for f in TELEMETRY_FIELDS]


class TelemetryJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        # msg 1: delivered, queue grows across its 3 fragments.
        rows += [_packet("e2g", DETECT_REQUEST, 1, i, 3, q, False) for i, q in enumerate((0.0, 1.0, 2.0))]
        rows += [_packet("g2e", DETECT_RESPONSE, 1, 0, 1, 0.0, False)]
        # msg 2: one request fragment lost on the link.
        rows += [_packet("e2g", DETECT_REQUEST, 2, i, 3, 0.5, i == 1) for i in range(3)]
        # msg 3: request delivered, reply lost.
        rows += [_packet("e2g", DETECT_REQUEST, 3, i, 3, 0.0, False) for i in range(3)]
        rows += [_packet("g2e", DETECT_RESPONSE, 3, 0, 1, 0.0, True)]
        # msg 4: everything delivered, but after the deadline.
        rows += [_packet("e2g", DETECT_REQUEST, 4, i, 3, 0.0, False) for i in range(3)]
        rows += [_packet("g2e", DETECT_RESPONSE, 4, 0, 1, 0.0, False)]
        self.tmp = tempfile.TemporaryDirectory()
        self.link_path = Path(self.tmp.name) / "link.csv"
        with open(self.link_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(TELEMETRY_FIELDS)
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_each_loss_is_attributed_to_its_cause(self) -> None:
        exchanges = [
            _exchange(1, "ok", arrival="1.04", rtt="80.0"),
            _exchange(2, "lost"),
            _exchange(3, "lost", arrival="3.04"),
            _exchange(4, "late", arrival="4.04", rtt="180.0"),
        ]
        rows = join(exchanges, load_link(self.link_path), deadline_ms=100.0)
        self.assertEqual([r["lost"] for r in rows], [0, 1, 1, 1])
        self.assertEqual(rows[0]["reason"], "-")
        self.assertEqual(rows[1]["reason"], "request: 1/3 pkts lost")
        self.assertEqual(rows[2]["reason"], "reply lost on link")
        self.assertEqual(rows[3]["reason"], "late (> 100 ms)")
        self.assertEqual(rows[0]["queue_delay_ms"], 2.0)  # the last fragment's wait
        self.assertIsNone(rows[1]["arrival_s"])

    def test_without_a_link_log_queue_delay_is_unknown(self) -> None:
        rows = join([_exchange(1, "ok", arrival="1.04")], None, deadline_ms=100.0)
        self.assertIsNone(rows[0]["queue_delay_ms"])
        self.assertEqual(rows[0]["packets_dropped"], 0)


if __name__ == "__main__":
    unittest.main()
