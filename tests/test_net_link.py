from __future__ import annotations

import csv
import io
import math
import socket
import struct
import threading
import time
import unittest

from jmdst.net import GilbertElliott, LinkConfig, LinkModel, LinkRelay, make_udp_socket, parse_netem


class ParseNetemTests(unittest.TestCase):
    def test_full_spec(self) -> None:
        cfg = parse_netem("delay 50ms 10ms distribution normal rate 10mbit loss random 1.5% limit 200 seed 7")
        self.assertEqual(cfg.delay_ms, 50.0)
        self.assertEqual(cfg.jitter_ms, 10.0)
        self.assertEqual(cfg.jitter_distribution, "normal")
        self.assertEqual(cfg.rate_bps, 10e6)
        self.assertEqual(cfg.loss_pct, 1.5)
        self.assertEqual(cfg.limit_packets, 200)
        self.assertEqual(cfg.seed, 7)

    def test_units(self) -> None:
        self.assertEqual(parse_netem("delay 0.2s").delay_ms, 200.0)
        self.assertEqual(parse_netem("delay 500us").delay_ms, 0.5)
        self.assertEqual(parse_netem("rate 256kbit").rate_bps, 256e3)
        # tc's *bps units are bytes per second.
        self.assertEqual(parse_netem("rate 1mbps").rate_bps, 8e6)

    def test_gemodel_defaults_follow_netem(self) -> None:
        ge = parse_netem("loss gemodel 2%").gilbert_elliott
        self.assertEqual((ge.p_pct, ge.r_pct, ge.bad_loss_pct, ge.good_loss_pct), (2.0, 98.0, 100.0, 0.0))
        ge = parse_netem("loss gemodel 1% 25% 80% 0.5%").gilbert_elliott
        self.assertEqual((ge.p_pct, ge.r_pct, ge.bad_loss_pct, ge.good_loss_pct), (1.0, 25.0, 80.0, 0.5))

    def test_empty_is_perfect_link(self) -> None:
        self.assertEqual(parse_netem(""), LinkConfig())

    def test_rejects_ambiguous_or_unknown_input(self) -> None:
        for bad in ("delay 50", "rate 10", "loss 5", "duplicate 1%", "delay"):
            with self.assertRaises(ValueError, msg=bad):
                parse_netem(bad)

    def test_round_trip_through_to_netem(self) -> None:
        for spec in ("delay 50ms 10ms rate 10mbit loss random 1%", "loss gemodel 1% 25% 100% 0%", "delay 5ms limit 50 seed 3"):
            cfg = parse_netem(spec)
            self.assertEqual(parse_netem(cfg.to_netem()), cfg)


class LinkModelTests(unittest.TestCase):
    def test_perfect_link_releases_immediately(self) -> None:
        model = LinkModel(LinkConfig())
        fate = model.offer(1.0, 1200)
        self.assertEqual(fate.release, 1.0)
        self.assertEqual(fate.queue_delay, 0.0)

    def test_serialisation_and_queueing_delay(self) -> None:
        # 8 Mbit/s and 1000-byte packets (no header) -> 1 ms each on the wire.
        model = LinkModel(LinkConfig(rate_bps=8e6, overhead_bytes=0, delay_ms=10.0))
        fates = [model.offer(0.0, 1000) for _ in range(3)]
        for i, fate in enumerate(fates):
            self.assertAlmostEqual(fate.queue_delay, i * 1e-3)
            self.assertAlmostEqual(fate.tx_end, (i + 1) * 1e-3)
            self.assertAlmostEqual(fate.release, (i + 1) * 1e-3 + 0.010)

    def test_overhead_counts_toward_airtime(self) -> None:
        model = LinkModel(LinkConfig(rate_bps=8e6))  # default 28-byte IPv4+UDP header
        self.assertAlmostEqual(model.offer(0.0, 972).tx_end, 1e-3)

    def test_tail_drop_when_transmit_buffer_full(self) -> None:
        model = LinkModel(LinkConfig(rate_bps=8e6, overhead_bytes=0, limit_packets=2))
        fates = [model.offer(0.0, 1000) for _ in range(5)]
        # First transmits at once; the next two wait (buffer = 2); rest dropped.
        self.assertEqual([f.drop for f in fates], [None, None, None, "queue", "queue"])
        # After the first waiting packet starts (t = 1 ms) there is room again.
        later = model.offer(1.5e-3, 1000)
        self.assertIsNone(later.drop)
        self.assertAlmostEqual(later.queue_delay, 3e-3 - 1.5e-3)

    def test_fifo_never_reorders_under_jitter(self) -> None:
        model = LinkModel(LinkConfig(delay_ms=50.0, jitter_ms=40.0, seed=1))
        releases = [model.offer(i * 1e-3, 100).release for i in range(2000)]
        self.assertEqual(releases, sorted(releases))

    def test_normal_jitter_cannot_precede_transmission(self) -> None:
        model = LinkModel(LinkConfig(delay_ms=1.0, jitter_ms=50.0, jitter_distribution="normal"))
        for i in range(2000):
            fate = model.offer(i * 0.2, 100)  # spaced so FIFO never binds
            self.assertGreaterEqual(fate.release, fate.tx_end)

    def test_uniform_jitter_bounds_and_spread(self) -> None:
        model = LinkModel(LinkConfig(delay_ms=50.0, jitter_ms=20.0, seed=2))
        delays = [model.offer(i * 1.0, 100).release - i * 1.0 for i in range(20000)]
        self.assertGreaterEqual(min(delays), 0.030 - 1e-12)
        self.assertLessEqual(max(delays), 0.070 + 1e-12)
        mean = sum(delays) / len(delays)
        std = math.sqrt(sum((d - mean) ** 2 for d in delays) / len(delays))
        self.assertAlmostEqual(mean, 0.050, delta=0.0005)
        self.assertAlmostEqual(std, 0.020 / math.sqrt(3), delta=0.0005)

    def test_seed_reproducibility(self) -> None:
        def run(seed: int) -> list:
            model = LinkModel(LinkConfig(delay_ms=20.0, jitter_ms=5.0, loss_pct=10.0, seed=seed))
            return [(f.drop, f.release) for f in (model.offer(i * 1e-3, 100) for i in range(500))]

        self.assertEqual(run(11), run(11))
        self.assertNotEqual(run(11), run(12))

    def test_channel_loss_pattern_independent_of_other_settings(self) -> None:
        # The i-th packet meets the same channel whatever the delay/jitter/rate,
        # so configurations can be compared packet-for-packet.
        def lost(cfg: LinkConfig) -> list[int]:
            model = LinkModel(cfg)
            return [i for i in range(3000) if model.offer(i * 0.01, 500).drop == "channel"]

        base = lost(LinkConfig(loss_pct=10.0, seed=5))
        self.assertEqual(base, lost(LinkConfig(loss_pct=10.0, seed=5, delay_ms=80.0, jitter_ms=30.0)))
        self.assertEqual(base, lost(LinkConfig(loss_pct=10.0, seed=5, rate_bps=1e6)))

    def test_bernoulli_mean_loss(self) -> None:
        model = LinkModel(LinkConfig(loss_pct=5.0, seed=3))
        n = 100_000
        drops = sum(model.offer(i * 1e-4, 100).drop is not None for i in range(n))
        self.assertAlmostEqual(100.0 * drops / n, 5.0, delta=0.25)  # ~3.6 sigma

    def test_gilbert_elliott_mean_loss_and_burst_length(self) -> None:
        ge = GilbertElliott(p_pct=1.0, r_pct=25.0)
        self.assertAlmostEqual(ge.mean_loss_pct, 100.0 / 26.0)
        self.assertAlmostEqual(ge.mean_burst_packets, 4.0)
        model = LinkModel(LinkConfig(gilbert_elliott=ge, seed=4))
        lost = [model.offer(i * 1e-4, 100).drop is not None for i in range(200_000)]
        self.assertAlmostEqual(100.0 * sum(lost) / len(lost), ge.mean_loss_pct, delta=0.4)
        bursts, run = [], 0
        for flag in lost:
            if flag:
                run += 1
            elif run:
                bursts.append(run)
                run = 0
        self.assertAlmostEqual(sum(bursts) / len(bursts), 4.0, delta=0.3)

    def test_counters(self) -> None:
        model = LinkModel(LinkConfig(rate_bps=8e6, overhead_bytes=0, limit_packets=1, loss_pct=0.0))
        for _ in range(4):
            model.offer(0.0, 1000)
        c = model.counters
        self.assertEqual((c.offered, c.delivered, c.dropped_queue), (4, 2, 2))
        self.assertEqual(c.payload_bytes_delivered, 2000)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            LinkConfig(loss_pct=1.0, gilbert_elliott=GilbertElliott(1.0, 25.0))
        with self.assertRaises(ValueError):
            LinkConfig(rate_bps=0.0)


class RelaySocketTests(unittest.TestCase):
    """Real UDP over localhost: the relay must add the modelled delay, drop what
    the model drops, and carry replies back to the edge."""

    def _start(self, e2g: LinkConfig, g2e: LinkConfig) -> tuple[LinkRelay, socket.socket, socket.socket, threading.Event, threading.Thread, io.StringIO]:
        ground = make_udp_socket()
        log = io.StringIO()
        relay = LinkRelay(("127.0.0.1", 0), ground.getsockname(), e2g, g2e, telemetry=log)
        edge = make_udp_socket()
        stop = threading.Event()
        thread = threading.Thread(target=relay.run, args=(stop,), daemon=True)
        thread.start()
        self.addCleanup(lambda: (stop.set(), thread.join(2), relay.close(), edge.close(), ground.close()))
        return relay, edge, ground, stop, thread, log

    def test_delay_is_applied_and_replies_return(self) -> None:
        relay, edge, ground, *_ = self._start(LinkConfig(delay_ms=20.0), LinkConfig(delay_ms=10.0))
        ground.settimeout(2.0)
        edge.settimeout(2.0)
        one_way, round_trip = [], []
        for i in range(10):
            sent = time.perf_counter()
            edge.sendto(struct.pack("<Id", i, sent), relay.edge_address)
            payload, addr = ground.recvfrom(2048)
            one_way.append(time.perf_counter() - struct.unpack("<Id", payload)[1])
            ground.sendto(payload, addr)  # reply to the sender, i.e. the relay
            edge.recvfrom(2048)
            round_trip.append(time.perf_counter() - sent)
        for d in one_way:
            self.assertGreaterEqual(d, 0.020)
            self.assertLess(d, 0.020 + 0.015)
        for d in round_trip:
            self.assertGreaterEqual(d, 0.030)

    def test_total_loss_delivers_nothing_and_is_logged(self) -> None:
        relay, edge, ground, stop, thread, log = self._start(LinkConfig(loss_pct=100.0), LinkConfig())
        ground.settimeout(0.3)
        for i in range(5):
            edge.sendto(bytes(100), relay.edge_address)
        with self.assertRaises(socket.timeout):
            ground.recvfrom(2048)
        stop.set()
        thread.join(2)
        self.assertEqual(relay.stats()["e2g"]["dropped_channel"], 5)
        rows = list(csv.DictReader(io.StringIO(log.getvalue())))
        self.assertEqual(len(rows), 5)  # one row per packet
        for row in rows:
            self.assertEqual((row["dropped"], row["drop_reason"]), ("1", "channel"))
            self.assertEqual(row["head"], "00" * 12)  # first 12 payload bytes, as a capture would show


if __name__ == "__main__":
    unittest.main()
