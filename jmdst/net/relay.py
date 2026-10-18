"""UDP relay that applies a ``LinkModel`` to each direction between two nodes.

This is the running equivalent of a netem bridge: the edge node sends to the
relay's edge-facing port, the relay forwards to the ground node, and the
ground node simply replies to whoever sent the packet (the relay), which
forwards the reply back to the edge. Neither node knows the link exists --
pointing the edge straight at the ground (or at a real network) needs no code
change, only a different address.

The relay is payload-agnostic, like netem: one UDP datagram is one packet.
Fragmenting messages into MTU-sized datagrams is the nodes' job, which is what
makes per-packet loss compound over a large frame the way it does on a real
link.

Timing on Windows
-----------------
Packets are held in a release-time heap and sent when due. Waking at the
right moment needs the system timer at 1 ms: at Windows' default resolution
``select`` overshoots by a median 10.9 ms (measured), which would swamp any
configured delay below ~30 ms. ``run`` raises the resolution for its duration
(median overshoot 0.46 ms, max 1.3 ms measured).

That alone is **not enough on Windows 11**: a windowless process that sits
mostly idle is classed as background after ~2 s and its resolution request is
silently ignored -- measured overshoot then climbs to a median ~12 ms. Busy
traffic hides this (frequent packets keep waking the loop); sparse traffic,
such as one detection frame every 100 ms, exposes it. ``high_resolution_timer``
therefore also opts the process out of that throttling
(``PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION``), which held 0.3-0.7 ms
medians over a 12 s idle test.

Every delivered packet's actual-minus-scheduled release time is kept in
``release_errors_ms`` so each run reports its own timing accuracy rather than
assuming it.
"""

from __future__ import annotations

import contextlib
import csv
import heapq
import itertools
import select
import signal
import socket
import sys
import threading
import time
from typing import Callable, Iterator, TextIO

import numpy as np

from .link import LinkConfig, LinkModel, PacketFate

# The largest val/test frame (2720x1530) is 12.5 MB as raw RGB -- ~9,000
# datagrams in one burst when a lossless codec is used. Windows grants 64 MB.
SOCKET_BUFFER_BYTES = 64 * 1024 * 1024
MAX_DATAGRAM_BYTES = 65535
DRAIN_BATCH = 64  # datagrams read per pass before releasing due packets
DIRECTIONS = ("e2g", "g2e")  # edge -> ground, ground -> edge

TELEMETRY_FIELDS = [
    "direction", "seq", "arrival_s", "payload_bytes", "queue_delay_ms",
    "scheduled_release_s", "actual_release_s", "release_error_ms", "dropped", "drop_reason", "head",
]
# Bytes of each payload copied into the log as hex, like a packet capture's
# snap length. The relay never interprets them; they let a report join packets
# to the protocol messages they carried (scripts/sim_telemetry.py).
LOG_HEAD_BYTES = 12


@contextlib.contextmanager
def high_resolution_timer(period_ms: int = 1) -> Iterator[None]:
    """Raise the Windows system timer resolution while the block runs.

    Also opts this process out of Windows 11's background power throttling:
    of timer requests (see module docstring) -- without it the request lapses
    after ~2 s of mostly-idle waiting -- and of execution speed (EcoQoS),
    which can move a windowless process to slower cores and clocks. Every
    simulation process uses this, so the only limits on a node are the ones
    the simulation sets deliberately.
    """

    if sys.platform != "win32":
        yield
        return
    import ctypes
    import ctypes.wintypes as wt

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    # ProcessPowerThrottling = 4. Control EXECUTION_SPEED (0x1) and
    # IGNORE_TIMER_RESOLUTION (0x4) with state 0 = "never throttle": no
    # EcoQoS demotion to slow cores/clocks, and timer requests honoured.
    # Best effort: older Windows lacks the class.
    state = _PowerThrottlingState(1, 0x1 | 0x4, 0)
    kernel32.SetProcessInformation(wt.HANDLE(kernel32.GetCurrentProcess()), 4, ctypes.byref(state), ctypes.sizeof(state))

    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(period_ms)
    try:
        yield
    finally:
        winmm.timeEndPeriod(period_ms)


def interrupt_on_ctrl_break() -> None:
    """Turn CTRL_BREAK into KeyboardInterrupt.

    On Windows a parent stops a console child (``scripts/run_sim.py`` stopping
    the ground and link processes) by sending CTRL_BREAK, which by default
    kills Python outright -- no summary, no flushed telemetry. With this, the
    child unwinds exactly as it would on Ctrl+C in its own terminal.
    """

    if hasattr(signal, "SIGBREAK"):
        def _raise(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGBREAK, _raise)


def make_udp_socket(bind: tuple[str, int] = ("127.0.0.1", 0)) -> socket.socket:
    """UDP socket with buffers large enough to absorb a fragmented frame burst.

    Windows' default 64 KB receive buffer silently drops most of a 100 KB
    frame sent as back-to-back datagrams -- loss the link model never chose.
    The nodes use these sockets too, for the same reason.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for option in (socket.SO_RCVBUF, socket.SO_SNDBUF):
        sock.setsockopt(socket.SOL_SOCKET, option, SOCKET_BUFFER_BYTES)
    sock.bind(bind)
    return sock


class LinkRelay:
    """Forward edge<->ground datagrams through one ``LinkModel`` per direction."""

    def __init__(
        self,
        edge_bind: tuple[str, int],
        ground_addr: tuple[str, int],
        edge_to_ground: LinkConfig,
        ground_to_edge: LinkConfig,
        telemetry: TextIO | None = None,
        clock: Callable[[], float] = time.perf_counter,
        ground_bind: tuple[str, int] = ("127.0.0.1", 0),
        log_head_bytes: int = LOG_HEAD_BYTES,
    ) -> None:
        self.edge_sock = make_udp_socket(edge_bind)
        self.ground_sock = make_udp_socket(ground_bind)
        for sock in (self.edge_sock, self.ground_sock):
            sock.setblocking(False)
        self.ground_addr = ground_addr
        self.edge_addr: tuple[str, int] | None = None  # learned from the first edge packet
        self.models = {"e2g": LinkModel(edge_to_ground), "g2e": LinkModel(ground_to_edge)}
        self.host_drops = {d: 0 for d in DIRECTIONS}
        self.release_errors_ms: list[float] = []
        self._seq = {d: itertools.count() for d in DIRECTIONS}
        self._clock = clock
        self._heap: list = []
        self._tiebreak = itertools.count()
        self._writer = csv.writer(telemetry) if telemetry is not None else None
        self._log_head_bytes = log_head_bytes
        if self._writer is not None:
            self._writer.writerow(TELEMETRY_FIELDS)

    @property
    def edge_address(self) -> tuple[str, int]:
        """Where the edge node should send."""

        return self.edge_sock.getsockname()

    def run(self, stop: threading.Event | None = None, duration: float | None = None) -> None:
        """Relay until ``stop`` is set or ``duration`` seconds pass.

        Safe to call repeatedly: queued and in-flight packets persist between
        calls, so a caller can run in slices and print stats in between.
        """

        with high_resolution_timer():
            deadline = None if duration is None else self._clock() + duration
            while stop is None or not stop.is_set():
                now = self._clock()
                if deadline is not None and now >= deadline:
                    return
                timeout = 0.05  # bounds how long a stop request can go unnoticed
                if self._heap:
                    timeout = min(timeout, max(0.0, self._heap[0][0] - now))
                readable, _, _ = select.select([self.edge_sock, self.ground_sock], [], [], timeout)
                for sock in readable:
                    self._drain(sock)
                self._release_due()

    def stats(self) -> dict:
        out: dict = {}
        in_flight = {d: 0 for d in DIRECTIONS}
        for entry in self._heap:
            in_flight[entry[2]] += 1
        for direction in DIRECTIONS:
            c = self.models[direction].counters
            out[direction] = {
                "offered": c.offered,
                "delivered": c.delivered - in_flight[direction] - self.host_drops[direction],
                "in_flight": in_flight[direction],
                "dropped_queue": c.dropped_queue,
                "dropped_channel": c.dropped_channel,
                "dropped_host": self.host_drops[direction],
                "payload_bytes_offered": c.payload_bytes_offered,
            }
        if self.release_errors_ms:
            errors = np.asarray(self.release_errors_ms)
            out["release_error_ms"] = {
                "p50": float(np.percentile(errors, 50)),
                "p95": float(np.percentile(errors, 95)),
                "max": float(errors.max()),
            }
        return out

    def close(self) -> None:
        self.edge_sock.close()
        self.ground_sock.close()

    # -- internals ----------------------------------------------------------

    def _drain(self, sock: socket.socket) -> None:
        # Read at most DRAIN_BATCH datagrams, then return so due packets get
        # released. Draining a whole burst first (a raw frame is ~1,200
        # datagrams) held the burst's head back until its tail was read --
        # measured 9 ms median release lateness on a *perfect* link. ``run``
        # comes straight back here because the socket is still readable.
        for _ in range(DRAIN_BATCH):
            try:
                payload, addr = sock.recvfrom(MAX_DATAGRAM_BYTES)
            except BlockingIOError:
                return
            except ConnectionResetError:
                # Windows surfaces an ICMP port-unreachable from an *earlier*
                # sendto (a node not yet listening) on the next recvfrom. It
                # says nothing about this socket; keep draining.
                continue
            now = self._clock()
            if sock is self.edge_sock:
                self.edge_addr = addr
                direction, out = "e2g", (self.ground_sock, self.ground_addr)
            else:
                direction, out = "g2e", (self.edge_sock, self.edge_addr)
            seq = next(self._seq[direction])
            fate = self.models[direction].offer(now, len(payload))
            if fate.delivered and out[1] is not None:
                heapq.heappush(self._heap, (fate.release, next(self._tiebreak), direction, seq, fate, payload, out))
            elif fate.delivered:
                # Ground spoke before the edge ever did: nowhere to deliver.
                self.host_drops[direction] += 1
                self._log(direction, seq, fate, None, "host", payload)
            else:
                self._log(direction, seq, fate, None, fate.drop, payload)

    def _release_due(self) -> None:
        while self._heap and self._heap[0][0] <= self._clock():
            release, _, direction, seq, fate, payload, (sock, addr) = heapq.heappop(self._heap)
            try:
                sock.sendto(payload, addr)
            except OSError:
                # Send buffer full or address unreachable: the host, not the
                # modelled link, lost it. Counted separately so it is visible.
                self.host_drops[direction] += 1
                self._log(direction, seq, fate, None, "host", payload)
                continue
            actual = self._clock()
            self.release_errors_ms.append((actual - release) * 1e3)
            self._log(direction, seq, fate, actual, None, payload)

    def _log(self, direction: str, seq: int, fate: PacketFate, actual: float | None, drop: str | None,
             payload: bytes = b"") -> None:
        if self._writer is None:
            return
        scheduled = fate.release
        self._writer.writerow([
            direction,
            seq,
            f"{fate.arrival:.6f}",
            fate.payload_bytes,
            f"{fate.queue_delay * 1e3:.3f}",
            "" if scheduled is None else f"{scheduled:.6f}",
            "" if actual is None else f"{actual:.6f}",
            "" if actual is None or scheduled is None else f"{(actual - scheduled) * 1e3:.3f}",
            0 if drop is None else 1,
            drop or "",
            payload[: self._log_head_bytes].hex(),
        ])
