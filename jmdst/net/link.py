"""Network link model for the edge/ground simulation (netem-equivalent).

Why not ``tc netem`` itself
---------------------------
The simulation runs on a Windows host. netem is a Linux qdisc, so using it
would put one or both nodes inside a Linux VM whose clock is not the host's --
and the telemetry this simulation exists to produce (send time vs arrival
time, i.e. one-way latency) is only meaningful when both ends read one clock.
This module reproduces netem's model instead, in-process and seeded, so every
timestamp in a run comes from the same ``time.perf_counter`` (QPC on
Windows, measured to be shared across processes).

Model
-----
Each direction of the link is one ``LinkModel``. A packet offered at time
``t`` goes through, in order:

1. **Transmit buffer** (netem ``limit``). Packets waiting for the transmitter
   beyond ``limit_packets`` are tail-dropped (``drop == "queue"``).
2. **Transmitter** (netem ``rate``). Serialisation takes
   ``(payload + overhead) * 8 / rate`` seconds, one packet at a time.
   ``queue_delay`` is the wait before transmission starts.
3. **Channel loss** (netem ``loss random`` / ``loss gemodel``). Applied *after*
   transmission, so a lost packet still consumed airtime -- the radio-link
   behaviour. netem drops at enqueue instead; the difference only matters when
   a rate limit is also set.
4. **Propagation** (netem ``delay``/jitter). Released at
   ``tx_end + delay + jitter``.

Deliberate differences from netem, both in the direction of a radio link
rather than a kernel qdisc:

* **FIFO is preserved.** netem lets jitter reorder packets; real point-to-point
  links do not, so a packet is never released before its predecessor.
* **``limit`` counts only packets waiting to transmit**, not packets in
  propagation. netem counts both, which makes a long delay exhaust the limit.

Reproducibility: loss and jitter draw from separate seeded streams, one draw
of each per offered packet whatever its fate. So the i-th packet meets the
same channel under any rate, queue, or jitter setting -- configurations can be
compared packet-for-packet instead of through run-to-run noise.

Configs are written in netem's own syntax (``parse_netem``), so the same
strings can be applied with ``tc qdisc ... netem`` on Linux for a cross-check.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass

import numpy as np

IPV4_UDP_OVERHEAD_BYTES = 28


@dataclass(frozen=True)
class GilbertElliott:
    """Two-state bursty loss (netem ``loss gemodel p [r [1-h [1-k]]]``).

    Percentages per packet: ``p`` = P(good -> bad), ``r`` = P(bad -> good),
    ``bad_loss`` = loss probability in the bad state (netem's 1-h),
    ``good_loss`` = in the good state (netem's 1-k). netem's defaults: r =
    100 - p, bad_loss 100, good_loss 0.
    """

    p_pct: float
    r_pct: float
    bad_loss_pct: float = 100.0
    good_loss_pct: float = 0.0

    @property
    def mean_loss_pct(self) -> float:
        p, r = self.p_pct / 100.0, self.r_pct / 100.0
        bad_share = p / (p + r) if p + r > 0 else 0.0
        return 100.0 * (bad_share * self.bad_loss_pct / 100.0 + (1 - bad_share) * self.good_loss_pct / 100.0)

    @property
    def mean_burst_packets(self) -> float:
        """Mean bad-state sojourn, in packets (= mean loss burst when bad_loss is 100%)."""

        return math.inf if self.r_pct == 0 else 100.0 / self.r_pct


@dataclass(frozen=True)
class LinkConfig:
    """One direction of the link. Defaults are a perfect link."""

    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    jitter_distribution: str = "uniform"  # netem's default when no table is given
    rate_bps: float | None = None  # None = unlimited
    limit_packets: int = 1000  # netem's default limit
    loss_pct: float = 0.0
    gilbert_elliott: GilbertElliott | None = None
    overhead_bytes: int = IPV4_UDP_OVERHEAD_BYTES
    seed: int = 0

    def __post_init__(self) -> None:
        if self.delay_ms < 0 or self.jitter_ms < 0:
            raise ValueError("delay and jitter must be non-negative")
        if self.jitter_distribution not in ("uniform", "normal"):
            raise ValueError(f"unknown jitter distribution {self.jitter_distribution!r}")
        if self.rate_bps is not None and self.rate_bps <= 0:
            raise ValueError("rate must be positive")
        if self.limit_packets < 1:
            raise ValueError("limit must be at least 1 packet")
        if not 0.0 <= self.loss_pct <= 100.0:
            raise ValueError("loss must be within 0-100%")
        if self.loss_pct and self.gilbert_elliott is not None:
            raise ValueError("choose one loss model: random or gemodel")

    @property
    def mean_loss_pct(self) -> float:
        return self.gilbert_elliott.mean_loss_pct if self.gilbert_elliott else self.loss_pct

    def to_netem(self) -> str:
        """Render in netem syntax (inverse of ``parse_netem``)."""

        parts: list[str] = []
        if self.delay_ms or self.jitter_ms:
            parts.append(f"delay {_g(self.delay_ms)}ms")
            if self.jitter_ms:
                parts.append(f"{_g(self.jitter_ms)}ms")
                if self.jitter_distribution != "uniform":
                    parts.append(f"distribution {self.jitter_distribution}")
        if self.rate_bps is not None:
            parts.append(f"rate {_g(self.rate_bps / 1e3)}kbit")
        if self.loss_pct:
            parts.append(f"loss random {_g(self.loss_pct)}%")
        if self.gilbert_elliott:
            ge = self.gilbert_elliott
            parts.append(
                f"loss gemodel {_g(ge.p_pct)}% {_g(ge.r_pct)}% {_g(ge.bad_loss_pct)}% {_g(ge.good_loss_pct)}%"
            )
        if self.limit_packets != 1000:
            parts.append(f"limit {self.limit_packets}")
        if self.seed:
            parts.append(f"seed {self.seed}")
        return " ".join(parts) or "(perfect link)"


def _g(value: float) -> str:
    return f"{value:g}"


_TIME_UNITS = {"s": 1e3, "sec": 1e3, "ms": 1.0, "msec": 1.0, "us": 1e-3, "usec": 1e-3}
# tc's rate units: *bit are bits/s, *bps are BYTES/s (a classic tc trap).
_RATE_UNITS = {
    "bit": 1.0, "kbit": 1e3, "mbit": 1e6, "gbit": 1e9,
    "bps": 8.0, "kbps": 8e3, "mbps": 8e6, "gbps": 8e9,
}


def _parse_time_ms(token: str) -> float:
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-z]+)", token.lower())
    if not match or match.group(2) not in _TIME_UNITS:
        # tc reads a bare number as microseconds; requiring a unit avoids
        # silently configuring a delay 1000x smaller than intended.
        raise ValueError(f"time {token!r} needs a unit (s, ms, us)")
    return float(match.group(1)) * _TIME_UNITS[match.group(2)]


def _parse_rate_bps(token: str) -> float:
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-z]+)", token.lower())
    if not match or match.group(2) not in _RATE_UNITS:
        raise ValueError(f"rate {token!r} needs a unit (bit, kbit, mbit, gbit, or bytes: bps, kbps, mbps)")
    return float(match.group(1)) * _RATE_UNITS[match.group(2)]


def _is_pct(token: str) -> bool:
    return re.fullmatch(r"[0-9]*\.?[0-9]+%", token) is not None


def _parse_pct(token: str) -> float:
    if not _is_pct(token):
        raise ValueError(f"percentage {token!r} must end in % (e.g. 5%)")
    return float(token[:-1])


def parse_netem(spec: str) -> LinkConfig:
    """Parse a netem option string, e.g. ``"delay 50ms 10ms rate 10mbit loss 1%"``.

    Supported: ``delay TIME [JITTER] [distribution uniform|normal]``,
    ``rate RATE``, ``loss [random] PCT``,
    ``loss gemodel P [R [1-H [1-K]]]``, ``limit N``, ``seed N``.
    An empty string is a perfect link.
    """

    tokens = spec.split()
    kwargs: dict = {}
    i = 0

    def take() -> str:
        nonlocal i
        if i >= len(tokens):
            raise ValueError(f"netem spec ended early: {spec!r}")
        i += 1
        return tokens[i - 1]

    def peek() -> str | None:
        return tokens[i] if i < len(tokens) else None

    while i < len(tokens):
        keyword = take().lower()
        if keyword == "delay":
            kwargs["delay_ms"] = _parse_time_ms(take())
            nxt = peek()
            if nxt is not None and re.match(r"[0-9.]", nxt):
                kwargs["jitter_ms"] = _parse_time_ms(take())
            if peek() is not None and peek().lower() == "distribution":
                take()
                kwargs["jitter_distribution"] = take().lower()
        elif keyword == "rate":
            kwargs["rate_bps"] = _parse_rate_bps(take())
        elif keyword == "loss":
            if peek() is not None and peek().lower() == "gemodel":
                take()
                p = _parse_pct(take())
                values = [p, 100.0 - p, 100.0, 0.0]
                for slot in (1, 2, 3):
                    if peek() is not None and _is_pct(peek()):
                        values[slot] = _parse_pct(take())
                kwargs["gilbert_elliott"] = GilbertElliott(*values)
            else:
                if peek() is not None and peek().lower() == "random":
                    take()
                kwargs["loss_pct"] = _parse_pct(take())
        elif keyword == "limit":
            kwargs["limit_packets"] = int(take())
        elif keyword == "seed":
            kwargs["seed"] = int(take())
        else:
            raise ValueError(f"unsupported netem option {keyword!r} in {spec!r}")
    return LinkConfig(**kwargs)


@dataclass(frozen=True)
class PacketFate:
    """What the link did with one offered packet. Times in seconds."""

    arrival: float
    payload_bytes: int
    queue_delay: float  # waited for the transmitter
    tx_end: float  # serialisation finished (== tx start when unlimited)
    release: float | None  # delivered at; None when dropped
    drop: str | None  # None, "queue" (buffer full) or "channel" (lost on air)

    @property
    def delivered(self) -> bool:
        return self.drop is None


@dataclass
class LinkCounters:
    offered: int = 0
    delivered: int = 0
    dropped_queue: int = 0
    dropped_channel: int = 0
    payload_bytes_offered: int = 0
    payload_bytes_delivered: int = 0


class LinkModel:
    """Deterministic one-direction link. Feed packets in arrival-time order."""

    def __init__(self, config: LinkConfig) -> None:
        self.config = config
        loss_seq, jitter_seq = np.random.SeedSequence(config.seed).spawn(2)
        self._loss_rng = np.random.default_rng(loss_seq)
        self._jitter_rng = np.random.default_rng(jitter_seq)
        self._waiting: deque[float] = deque()  # tx start times of queued packets
        self._tx_free = -math.inf
        self._last_release = -math.inf
        self._ge_bad = False
        self.counters = LinkCounters()

    def offer(self, now: float, payload_bytes: int) -> PacketFate:
        cfg = self.config
        c = self.counters
        c.offered += 1
        c.payload_bytes_offered += payload_bytes

        # One draw from each stream per packet, regardless of its fate, so the
        # channel the i-th packet meets is independent of rate/queue settings.
        channel_lost = self._draw_channel_loss()
        jitter_ms = self._draw_jitter_ms()

        while self._waiting and self._waiting[0] <= now:
            self._waiting.popleft()

        if cfg.rate_bps is None:
            start = end = now
        else:
            if len(self._waiting) >= cfg.limit_packets:
                c.dropped_queue += 1
                return PacketFate(now, payload_bytes, 0.0, now, None, "queue")
            start = max(now, self._tx_free)
            end = start + (payload_bytes + cfg.overhead_bytes) * 8.0 / cfg.rate_bps
            self._tx_free = end
            if start > now:
                self._waiting.append(start)

        if channel_lost:
            c.dropped_channel += 1
            return PacketFate(now, payload_bytes, start - now, end, None, "channel")

        # Normal jitter can exceed the base delay; a packet still cannot arrive
        # before it has finished transmitting.
        release = max(end, end + (cfg.delay_ms + jitter_ms) / 1e3)
        release = max(release, self._last_release)  # FIFO: never overtake
        self._last_release = release
        c.delivered += 1
        c.payload_bytes_delivered += payload_bytes
        return PacketFate(now, payload_bytes, start - now, end, release, None)

    def _draw_channel_loss(self) -> bool:
        u = self._loss_rng.random()
        ge = self.config.gilbert_elliott
        if ge is None:
            return u < self.config.loss_pct / 100.0
        # Loss is decided by the state the packet arrives in, then the state
        # steps -- the same order as netem's loss_gilb_ell().
        lost = u < (ge.bad_loss_pct if self._ge_bad else ge.good_loss_pct) / 100.0
        v = self._loss_rng.random()
        if self._ge_bad:
            self._ge_bad = not (v < ge.r_pct / 100.0)
        else:
            self._ge_bad = v < ge.p_pct / 100.0
        return lost

    def _draw_jitter_ms(self) -> float:
        cfg = self.config
        if cfg.jitter_distribution == "normal":
            return float(self._jitter_rng.normal(0.0, 1.0)) * cfg.jitter_ms
        return float(self._jitter_rng.uniform(-1.0, 1.0)) * cfg.jitter_ms
