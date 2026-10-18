"""Edge and ground node logic for the split deployment.

The split follows JMDST's own seam. The **edge** (the UAV) runs the whole
tracker -- Kalman, FELNet, association -- and owns the tau schedule. Only the
detection branch's YOLO call leaves the aircraft: ``RemoteDetector`` is a
drop-in ``detector(image)`` callable that ships the frame to the **ground**
(``GroundServer``) and waits for the boxes. If they do not arrive by the
deadline it returns ``None``, and ``JMDSTTracker`` runs the tracking branch
for that frame instead -- the paper's missed-detection mechanism, now also
covering the network.

The edge waits synchronously: Algorithm 1's detection branch at frame t
uses frame t's detections, so applying a late reply to a later frame would
change the algorithm. A reply that misses its deadline is discarded and
recorded as ``late``; one that never comes is ``lost``.

Every exchange is recorded on the edge as an ``Exchange`` with one clock's
timestamps (edge send, ground arrival, ground compute, ground send, edge
arrival), so uplink/downlink/RTT come straight from the record.
"""

from __future__ import annotations

import itertools
import select
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, TextIO

import numpy as np

from .protocol import (
    CODECS, DEFAULT_MTU, DETECT_REQUEST, DETECT_RESPONSE, PING, PONG,
    DetectRequest, DetectResponse, Reassembler, decode_image, encode_image, fragment,
    pack_info, pack_request, pack_response, unpack_info, unpack_request, unpack_response,
)
from .relay import make_udp_socket

Clock = Callable[[], float]
GroundDetect = Callable[[object, float, float, int], list[tuple[np.ndarray, float]]]


@dataclass
class Exchange:
    """One detection frame's round trip, from the edge's point of view."""

    msg_id: int
    context: dict
    request_bytes: int  # datagram bytes on the wire, our headers included
    request_packets: int
    encode_ms: float
    send_s: float
    outcome: str = "pending"  # ok | lost | late
    ground_arrival_s: float | None = None
    ground_compute_ms: float | None = None
    response_send_s: float | None = None
    response_arrival_s: float | None = None
    response_bytes: int | None = None
    response_packets: int | None = None
    detections: int | None = None

    @property
    def uplink_ms(self) -> float | None:
        return None if self.ground_arrival_s is None else (self.ground_arrival_s - self.send_s) * 1e3

    @property
    def downlink_ms(self) -> float | None:
        if self.response_arrival_s is None or self.response_send_s is None:
            return None
        return (self.response_arrival_s - self.response_send_s) * 1e3

    @property
    def rtt_ms(self) -> float | None:
        return None if self.response_arrival_s is None else (self.response_arrival_s - self.send_s) * 1e3


EXCHANGE_FIELDS = [
    "sequence", "frame_id", "msg_id", "outcome", "request_bytes", "request_packets", "encode_ms",
    "send_s", "ground_arrival_s", "uplink_ms", "ground_compute_ms", "response_send_s",
    "response_arrival_s", "downlink_ms", "rtt_ms", "response_bytes", "response_packets", "detections",
]


def exchange_row(ex: Exchange) -> list:
    def f(value, digits=3):
        return "" if value is None else f"{value:.{digits}f}"

    return [
        ex.context.get("sequence", ""), ex.context.get("frame_id", ""), ex.msg_id, ex.outcome,
        ex.request_bytes, ex.request_packets, f(ex.encode_ms), f(ex.send_s, 6), f(ex.ground_arrival_s, 6),
        f(ex.uplink_ms), f(ex.ground_compute_ms), f(ex.response_send_s, 6), f(ex.response_arrival_s, 6),
        f(ex.downlink_ms), f(ex.rtt_ms), "" if ex.response_bytes is None else ex.response_bytes,
        "" if ex.response_packets is None else ex.response_packets,
        "" if ex.detections is None else ex.detections,
    ]


class RemoteDetector:
    """Edge-side ``detector(image)`` that asks the ground node over UDP.

    Returns the ground's detections, or ``None`` when no complete reply
    arrives within ``deadline_s`` of sending.
    """

    def __init__(
        self,
        ground_addr: tuple[str, int],
        conf: float,
        iou: float,
        imgsz: int,
        codec: str = "jpeg",
        jpeg_quality: int = 90,
        deadline_s: float = 0.1,
        mtu: int = DEFAULT_MTU,
        clock: Clock = time.perf_counter,
    ) -> None:
        if codec not in CODECS:
            raise ValueError(f"unknown codec {codec!r}")
        self.ground_addr = ground_addr
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.codec, self.jpeg_quality = codec, jpeg_quality
        self.deadline_s = deadline_s
        self.mtu = mtu
        self.sock = make_udp_socket()
        self.sock.setblocking(False)
        self.exchanges: list[Exchange] = []
        self.context: dict = {}  # copied into each Exchange (e.g. sequence, frame_id)
        self._clock = clock
        self._ids = itertools.count(1)
        self._reassembler = Reassembler()
        self._timed_out: dict[int, Exchange] = {}
        self.last_wait_ms = 0.0  # network wait of the most recent call (edge compute excluded)

    def handshake(self, timeout_s: float = 30.0, retry_s: float = 0.25) -> dict:
        """PING until the ground answers; returns its info. Raises on timeout."""

        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            msg_id = next(self._ids)
            for datagram in fragment(PING, msg_id, b"", self.mtu):
                self.sock.sendto(datagram, self.ground_addr)
            reply = self._wait_for(PONG, msg_id, self._clock() + retry_s)
            if reply is not None:
                return unpack_info(reply.body)
        raise TimeoutError(f"ground node at {self.ground_addr} did not answer within {timeout_s:.0f} s")

    def __call__(self, image) -> list[tuple[np.ndarray, float]] | None:
        t_enc = self._clock()
        image_data = encode_image(image, self.codec, self.jpeg_quality)
        encode_ms = (self._clock() - t_enc) * 1e3

        msg_id = next(self._ids)
        send_s = self._clock()
        body = pack_request(DetectRequest(
            int(self.context.get("frame_id", 0)), send_s, self.conf, self.iou, self.imgsz,
            CODECS[self.codec], image_data,
        ))
        datagrams = fragment(DETECT_REQUEST, msg_id, body, self.mtu)
        ex = Exchange(msg_id, dict(self.context), sum(map(len, datagrams)), len(datagrams), encode_ms, send_s)
        self.exchanges.append(ex)
        for datagram in datagrams:
            self.sock.sendto(datagram, self.ground_addr)

        reply = self._wait_for(DETECT_RESPONSE, msg_id, send_s + self.deadline_s)
        self.last_wait_ms = (self._clock() - send_s) * 1e3
        if reply is None:
            ex.outcome = "lost"  # upgraded to "late" if the reply turns up afterwards
            self._timed_out[msg_id] = ex
            return None
        resp = unpack_response(reply.body)
        self._fill(ex, reply, resp)
        ex.outcome = "ok"
        return resp.detections

    def drain(self, wait_s: float = 0.0) -> None:
        """Collect stragglers so late replies are classified before reporting."""

        self._wait_for(None, None, self._clock() + wait_s)

    def close(self) -> None:
        self.sock.close()

    # -- internals ----------------------------------------------------------

    def _fill(self, ex: Exchange, reply, resp: DetectResponse) -> None:
        ex.ground_arrival_s = resp.request_arrival_s
        ex.ground_compute_ms = resp.compute_ms
        ex.response_send_s = resp.response_send_s
        ex.response_arrival_s = reply.complete
        ex.response_bytes = reply.wire_bytes
        ex.response_packets = reply.fragments
        ex.detections = len(resp.detections)

    def _wait_for(self, msg_type: int | None, msg_id: int | None, deadline: float):
        """Receive until the wanted message completes or ``deadline`` passes."""

        while True:
            for datagram in self._recv_all():
                message = self._reassembler.add(datagram, self._clock())
                if message is None:
                    continue
                if message.msg_type == msg_type and message.msg_id == msg_id:
                    return message
                late = self._timed_out.pop(message.msg_id, None)
                if late is not None and message.msg_type == DETECT_RESPONSE:
                    self._fill(late, message, unpack_response(message.body))
                    late.outcome = "late"
            remaining = deadline - self._clock()
            if remaining <= 0:
                # Fragments of replies we have given up on would otherwise pile up.
                self._reassembler.expire(older_than=self._clock() - 5.0)
                return None
            select.select([self.sock], [], [], remaining)

    def _recv_all(self) -> list[bytes]:
        out = []
        while True:
            try:
                out.append(self.sock.recv(65535))
            except BlockingIOError:
                return out
            except ConnectionResetError:
                continue  # Windows: ICMP unreachable from an earlier send; not fatal


GROUND_FIELDS = [
    "msg_id", "frame_id", "outcome", "first_arrival_s", "complete_s", "request_bytes", "request_packets",
    "decode_ms", "detect_ms", "detections", "response_bytes", "response_packets", "source",
]


@dataclass
class GroundStats:
    served: int = 0
    incomplete: int = 0
    pings: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    detect_ms: list[float] = field(default_factory=list)


class GroundServer:
    """Serve detection requests: reassemble, decode, detect, reply to sender."""

    def __init__(
        self,
        bind: tuple[str, int],
        detect: GroundDetect,
        info: dict | None = None,
        mtu: int = DEFAULT_MTU,
        telemetry: TextIO | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        incomplete_after_s: float = 2.0,
        clock: Clock = time.perf_counter,
        cache=None,
    ) -> None:
        """``cache``: an optional ``DetectionCache`` -- the modelled ground
        station (see jmdst/net/detcache.py). Hits reply after the frame's
        recorded unloaded compute time; misses run ``detect`` live."""

        import csv

        self.sock = make_udp_socket(bind)
        self.sock.setblocking(False)
        self.detect = detect
        self.info = info or {}
        self.mtu = mtu
        self.incomplete_after_s = incomplete_after_s
        self.cache = cache
        self.stats = GroundStats()
        self._on_event = on_event or (lambda kind, data: None)
        self._clock = clock
        self._reassembler = Reassembler()
        self._writer = csv.writer(telemetry) if telemetry is not None else None
        if self._writer is not None:
            self._writer.writerow(GROUND_FIELDS)

    @property
    def address(self) -> tuple[str, int]:
        return self.sock.getsockname()

    def serve(self, stop=None, duration: float | None = None) -> None:
        deadline = None if duration is None else self._clock() + duration
        while stop is None or not stop.is_set():
            if deadline is not None and self._clock() >= deadline:
                return
            readable, _, _ = select.select([self.sock], [], [], 0.05)
            if readable:
                self._drain()
            for gone in self._reassembler.expire(older_than=self._clock() - self.incomplete_after_s):
                self.stats.incomplete += 1
                self._log([gone.msg_id, "", "incomplete", f"{gone.first_arrival:.6f}", "", "",
                           f"{gone.received}/{gone.expected}", "", "", "", "", "", ""])
                self._on_event("incomplete", {"msg_id": gone.msg_id, "received": gone.received, "expected": gone.expected})

    def close(self) -> None:
        self.sock.close()

    def _drain(self) -> None:
        while True:
            try:
                datagram, addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                return
            except ConnectionResetError:
                continue
            message = self._reassembler.add(datagram, self._clock())
            if message is None:
                continue
            if message.msg_type == PING:
                self.stats.pings += 1
                self._send(PONG, message.msg_id, pack_info(self.info), addr)
            elif message.msg_type == DETECT_REQUEST:
                self._serve_request(message, addr)

    def _serve_request(self, message, addr) -> None:
        req = unpack_request(message.body)
        hit = self.cache.lookup(req.image_data, req.conf, req.iou, req.imgsz) if self.cache is not None else None
        if hit is not None:
            # Modelled compute: reply once this frame's own unloaded
            # decode + detect time has passed since its last packet arrived.
            delay = message.complete + hit.compute_ms / 1e3 - self._clock()
            if delay > 0:
                time.sleep(delay)
            detections, decode_ms, detect_ms, source = hit.detections, None, hit.compute_ms, "cache"
            self.stats.cache_hits += 1
        else:
            t0 = self._clock()
            image = decode_image(req.codec, req.image_data)
            t1 = self._clock()
            detections = self.detect(image, req.conf, req.iou, req.imgsz)
            decode_ms, detect_ms = (t1 - t0) * 1e3, (self._clock() - t1) * 1e3
            source = "live" if self.cache is None else "live (cache miss)"
            if self.cache is not None:
                self.stats.cache_misses += 1
        resp_send = self._clock()
        body = pack_response(DetectResponse(
            req.frame_id, req.send_s, message.complete, (resp_send - message.complete) * 1e3, resp_send, detections,
        ))
        sent = self._send(DETECT_RESPONSE, message.msg_id, body, addr)
        self.stats.served += 1
        self.stats.detect_ms.append(detect_ms)
        self._log([
            message.msg_id, req.frame_id, "served", f"{message.first_arrival:.6f}", f"{message.complete:.6f}",
            message.wire_bytes, message.fragments, "" if decode_ms is None else f"{decode_ms:.3f}",
            f"{detect_ms:.3f}", len(detections), sum(map(len, sent)), len(sent), source,
        ])
        self._on_event("served", {
            "msg_id": message.msg_id, "frame_id": req.frame_id, "request_bytes": message.wire_bytes,
            "packets": message.fragments, "detections": len(detections), "detect_ms": detect_ms, "source": source,
        })

    def _send(self, msg_type: int, msg_id: int, body: bytes, addr) -> list[bytes]:
        datagrams = fragment(msg_type, msg_id, body, self.mtu)
        for datagram in datagrams:
            self.sock.sendto(datagram, addr)
        return datagrams

    def _log(self, row: list) -> None:
        if self._writer is not None:
            self._writer.writerow(row)
