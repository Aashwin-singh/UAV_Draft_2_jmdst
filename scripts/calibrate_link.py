"""Calibrate the link emulator: does the running relay do what it is told?

Sends real UDP traffic through ``LinkRelay`` (in its own process, as in a
real run) for a set of netem scenarios and checks, per scenario:

* **configured vs measured** -- mean one-way delay, jitter spread, loss rate,
  loss burst length, throughput -- against the analytical value of the
  configured parameters; and
* **packet-level agreement with the model** -- the same seeded ``LinkModel``,
  replayed offline on the recorded send times, predicts exactly which packets
  the channel loses and when each should arrive. Measured minus predicted
  arrival is the emulator's own timing error. (Buffer-overflow drops depend
  on sub-ms arrival timing, so under a rate limit the drop *rate* is checked
  against queueing theory instead.)

All timestamps come from one clock (``time.perf_counter``; QPC on Windows,
shared across processes), so one-way delay is measured directly rather than
inferred from round trips.

Usage:
    python scripts/calibrate_link.py                 # all scenarios, ~40 s
    python scripts/calibrate_link.py --quick         # fewer packets, ~15 s

Exits 1 if any check fails. Results: outputs/sim/link_calibration.json, plus a
per-packet relay log per scenario under outputs/sim/calibration/.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.net import LinkModel, LinkRelay, high_resolution_timer, make_udp_socket, parse_netem
from jmdst.net.link import IPV4_UDP_OVERHEAD_BYTES

HEADER = struct.Struct("<Id")  # sequence number, send time (perf_counter)
PAYLOAD_BYTES = 1200


@dataclass(frozen=True)
class Scenario:
    name: str
    spec: str
    packets: int
    interval_ms: float


def scenarios(quick: bool) -> list[Scenario]:
    k = 0.4 if quick else 1.0
    wire_bits = (PAYLOAD_BYTES + IPV4_UDP_OVERHEAD_BYTES) * 8
    return [
        Scenario("perfect link (floor)", "", int(1000 * k), 5.0),
        Scenario("fixed delay", "delay 50ms", int(500 * k), 5.0),
        # 50 ms spacing > the 40 ms jitter range, so FIFO never binds and the
        # spread is the pure uniform +-20 ms one.
        Scenario("delay + jitter", "delay 50ms 20ms", int(200 * k), 50.0),
        Scenario("random loss", "loss random 5% seed 1", int(5000 * k), 1.0),
        Scenario("bursty loss (GE)", "loss gemodel 1% 25% seed 2", int(5000 * k), 1.0),
        Scenario("rate, under capacity", "rate 10mbit", int(3000 * k), wire_bits / 8e6 * 1e3),  # offer 8 Mbit/s
        Scenario("rate, overloaded", "rate 10mbit limit 100", int(4000 * k), wire_bits / 15e6 * 1e3),  # offer 15
    ]


def _relay_process(spec: str, ground_addr, log_path: str, ready, stop, results) -> None:
    with open(log_path, "w", newline="", encoding="utf-8") as log:
        relay = LinkRelay(("127.0.0.1", 0), ground_addr, parse_netem(spec), parse_netem(""), telemetry=log)
        ready.put(relay.edge_address)
        relay.run(stop)
        results.put(relay.stats())
        relay.close()


def run_scenario(sc: Scenario, log_dir: Path) -> dict:
    cfg = parse_netem(sc.spec)
    ground = make_udp_socket()
    ground.settimeout(0.2)
    received: dict[int, tuple[float, float]] = {}  # seq -> (send, recv)
    done = threading.Event()

    def receive() -> None:
        while not done.is_set():
            try:
                data = ground.recv(65535)
            except TimeoutError:
                continue
            now = time.perf_counter()
            seq, sent = HEADER.unpack_from(data)
            received[seq] = (sent, now)

    ctx = mp.get_context("spawn")
    ready, stop, results = ctx.Queue(), ctx.Event(), ctx.Queue()
    log_path = log_dir / (sc.name.split(" (")[0].replace(" ", "_").replace(",", "") + ".csv")
    proc = ctx.Process(target=_relay_process, args=(sc.spec, ground.getsockname(), str(log_path), ready, stop, results))
    proc.start()
    relay_addr = ready.get(timeout=30)
    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()

    edge = make_udp_socket()
    padding = bytes(PAYLOAD_BYTES - HEADER.size)
    send_times = np.empty(sc.packets)
    interval = sc.interval_ms / 1e3
    # Send every packet that is due, then *sleep* to the next one. Never spin:
    # a spinning sender holds the GIL and delays the receiver thread's
    # timestamps by up to the 5 ms switch interval, which would be charged to
    # the emulator.
    t0 = time.perf_counter() + 0.05
    i = 0
    while i < sc.packets:
        now = time.perf_counter()
        while i < sc.packets and t0 + i * interval <= now:
            send_times[i] = time.perf_counter()
            edge.sendto(HEADER.pack(i, send_times[i]) + padding, relay_addr)
            i += 1
        if i < sc.packets:
            time.sleep(max(0.0, t0 + i * interval - time.perf_counter()))

    # Let everything in flight land: delay + jitter + a full buffer's drain.
    drain = (cfg.delay_ms + 3 * cfg.jitter_ms) / 1e3 + 0.5
    if cfg.rate_bps:
        drain += cfg.limit_packets * (PAYLOAD_BYTES + IPV4_UDP_OVERHEAD_BYTES) * 8 / cfg.rate_bps
    time.sleep(drain)
    stop.set()
    relay_stats = results.get(timeout=30)
    proc.join(10)
    done.set()
    receiver.join(2)
    edge.close()
    ground.close()
    return evaluate(sc, cfg, send_times, received, relay_stats)


def evaluate(sc: Scenario, cfg, send_times: np.ndarray, received: dict, relay_stats: dict) -> dict:
    n = len(send_times)
    lost_mask = np.array([i not in received for i in range(n)])
    delays_ms = np.array([(received[i][1] - received[i][0]) * 1e3 for i in sorted(received)])
    recv_times = np.array([received[i][1] for i in sorted(received)])

    # Replay the identical seeded model on the recorded send times.
    model = LinkModel(cfg)
    fates = [model.offer(t, PAYLOAD_BYTES) for t in send_times]
    predicted_lost = np.array([not f.delivered for f in fates])
    agreement = float((predicted_lost == lost_mask).mean())
    timing_error_ms = np.array([(received[i][1] - fates[i].release) * 1e3 for i in sorted(received) if fates[i].delivered])

    checks: list[dict] = []

    def check(metric: str, configured: float, measured: float, ok: bool) -> None:
        checks.append({"metric": metric, "configured": configured, "measured": measured, "ok": bool(ok)})

    loss_pct = 100.0 * lost_mask.mean()
    p = cfg.mean_loss_pct / 100.0
    if cfg.rate_bps is None:
        # Binomial 4-sigma band (GE losses are correlated, so widen it 3x).
        sigma = 100.0 * math.sqrt(max(p * (1 - p), 1e-9) / n) * (3.0 if cfg.gilbert_elliott else 1.0)
        check("loss %", cfg.mean_loss_pct, loss_pct, abs(loss_pct - cfg.mean_loss_pct) <= 4 * sigma + 0.05)
    if cfg.gilbert_elliott is not None:
        runs, run = [], 0
        for flag in lost_mask:
            if flag:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        burst = float(np.mean(runs)) if runs else 0.0
        check("mean loss burst (packets)", cfg.gilbert_elliott.mean_burst_packets, burst,
              abs(burst - cfg.gilbert_elliott.mean_burst_packets) <= 0.3 * cfg.gilbert_elliott.mean_burst_packets)
    if cfg.rate_bps is None and len(delays_ms):
        # Floor of 0-2 ms (loopback + relay wake-up), widened by the sampling
        # error of the mean when jitter makes the delay random.
        mean = float(delays_ms.mean())
        se = cfg.jitter_ms / math.sqrt(3.0) / math.sqrt(len(delays_ms))
        check("mean one-way delay ms", cfg.delay_ms, mean, -4 * se <= mean - cfg.delay_ms <= 2.0 + 4 * se)
    if cfg.jitter_ms and len(delays_ms) > 1:
        expected_sd = cfg.jitter_ms / math.sqrt(3.0)  # uniform +-J
        sd = float(delays_ms.std())
        check("delay std dev ms", expected_sd, sd, abs(sd - expected_sd) <= 0.15 * expected_sd)
    if cfg.rate_bps is not None and len(recv_times) > 10:
        bits = (len(recv_times) - 1) * (PAYLOAD_BYTES + IPV4_UDP_OVERHEAD_BYTES) * 8
        mbps = bits / (recv_times[-1] - recv_times[0]) / 1e6
        offered = (PAYLOAD_BYTES + IPV4_UDP_OVERHEAD_BYTES) * 8 / (sc.interval_ms / 1e3) / 1e6
        expected = min(offered, cfg.rate_bps / 1e6)
        check("throughput Mbit/s (wire)", expected, mbps, abs(mbps - expected) <= 0.03 * expected)
        if offered > cfg.rate_bps / 1e6:
            # A full buffer holds limit packets; each waits behind the rest.
            tx_ms = (PAYLOAD_BYTES + IPV4_UDP_OVERHEAD_BYTES) * 8 / cfg.rate_bps * 1e3
            steady = float(np.median(delays_ms[len(delays_ms) // 2:]))
            expected_q = (cfg.limit_packets + 1) * tx_ms + cfg.delay_ms
            check("steady-state delay ms (full buffer)", expected_q, steady, abs(steady - expected_q) <= 0.05 * expected_q + 1.0)
    if cfg.rate_bps is None:
        # Channel loss is a seeded per-packet draw, so the very same packets
        # must be lost as in the offline replay.
        check("loss pattern matches model %", 100.0, 100.0 * agreement, agreement >= 0.99)
    else:
        # Buffer overflow is not: whether a packet finds room depends on its
        # sub-ms arrival time at the relay versus its send time here. Check the
        # drop *rate* against queueing theory instead -- everything the link
        # can carry while sending, plus the buffer it drains afterwards.
        wire_bits = (PAYLOAD_BYTES + IPV4_UDP_OVERHEAD_BYTES) * 8
        carried = n * sc.interval_ms / 1e3 * cfg.rate_bps / wire_bits + cfg.limit_packets + 1
        expected_drop = 100.0 * max(0.0, n - carried) / n
        check("buffer-overflow drop %", expected_drop, loss_pct, abs(loss_pct - expected_drop) <= 1.5)
    if len(timing_error_ms):
        p95 = float(np.percentile(timing_error_ms, 95))
        check("timing error vs model p95 ms", 0.0, p95, p95 <= 2.0)

    return {
        "scenario": sc.name,
        "netem": cfg.to_netem(),
        "packets": n,
        "received": len(received),
        "delay_ms": None if not len(delays_ms) else {
            "mean": float(delays_ms.mean()), "p50": float(np.percentile(delays_ms, 50)),
            "p95": float(np.percentile(delays_ms, 95)), "max": float(delays_ms.max()),
        },
        "timing_error_ms": None if not len(timing_error_ms) else {
            "p50": float(np.percentile(timing_error_ms, 50)), "p95": float(np.percentile(timing_error_ms, 95)),
            "max": float(timing_error_ms.max()),
        },
        "relay": relay_stats,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Fewer packets per scenario.")
    parser.add_argument("--output", default="outputs/sim/link_calibration.json")
    args = parser.parse_args(argv)

    out_path = PROJECT_ROOT / args.output
    log_dir = out_path.parent / "calibration"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Link emulator calibration -- one clock (perf_counter) for every timestamp\n")
    print(f"{'scenario':24s} {'metric':36s} {'configured':>11s} {'measured':>10s}  result")
    print("-" * 92)
    results = []
    with high_resolution_timer():
        for sc in scenarios(args.quick):
            result = run_scenario(sc, log_dir)
            results.append(result)
            for i, c in enumerate(result["checks"]):
                label = sc.name if i == 0 else ""
                print(f"{label:24s} {c['metric']:36s} {c['configured']:11.2f} {c['measured']:10.2f}  {'ok' if c['ok'] else 'FAIL'}")
            print()

    failures = [(r["scenario"], c["metric"]) for r in results for c in r["checks"] if not c["ok"]]
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    total = sum(len(r["checks"]) for r in results)
    print(f"{total - len(failures)}/{total} checks passed. Results: {out_path.relative_to(PROJECT_ROOT)}")
    for scenario, metric in failures:
        print(f"  FAIL: {scenario} -- {metric}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
