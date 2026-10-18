"""Wire format between the edge and ground nodes.

Messages are split into datagrams of at most ``mtu`` bytes, each carrying a
12-byte header. There is **no retransmission**: like a real-time video link
over RTP/UDP, a message with any fragment missing is lost. That is what
makes per-packet loss compound over a large frame -- at 1% packet loss a
75-packet JPEG frame arrives intact only 0.99**75 = 47% of the time -- and it
is the behaviour the evaluation needs to see, not something to hide.

Message types
-------------
* ``DETECT_REQUEST`` (edge -> ground): one detection frame, plus the
  post-processing the edge wants (conf, NMS IoU, imgsz). The edge is the only
  place those are configured, so the two nodes cannot disagree about them.
* ``DETECT_RESPONSE`` (ground -> edge): the boxes as float64 (bit-exact with
  the single-process pipeline), plus the ground's timestamps, so the edge
  alone holds a complete, single-clock record of every exchange.
* ``PING`` / ``PONG``: start-up handshake, so an edge never starts while the
  ground is down and silently runs a whole sequence on the tracking branch.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

MAGIC = b"JM"
VERSION = 1
DETECT_REQUEST, DETECT_RESPONSE, PING, PONG = 1, 2, 3, 4

FRAGMENT_HEADER = struct.Struct("<2sBBIHH")  # magic, version, type, msg_id, index, count
REQUEST_META = struct.Struct("<IdddHB")  # frame_id, send_s, conf, iou, imgsz, codec
RESPONSE_META = struct.Struct("<IddddI")  # frame_id, req_send_s, req_arrival_s, compute_ms, resp_send_s, n
DEFAULT_MTU = 1400  # datagram payload bytes incl. our header; safely under 1472

CODECS = {"raw": 0, "png": 1, "jpeg": 2}
_CODEC_NAMES = {v: k for k, v in CODECS.items()}
_RAW_SIZE = struct.Struct("<HH")


# -- fragmentation ------------------------------------------------------------

def fragment(msg_type: int, msg_id: int, body: bytes, mtu: int = DEFAULT_MTU) -> list[bytes]:
    chunk = mtu - FRAGMENT_HEADER.size
    if chunk <= 0:
        raise ValueError(f"mtu {mtu} leaves no room for payload")
    count = max(1, -(-len(body) // chunk))
    if count > 0xFFFF:
        raise ValueError(f"message of {len(body)} bytes needs {count} fragments (max 65535)")
    return [
        FRAGMENT_HEADER.pack(MAGIC, VERSION, msg_type, msg_id, i, count) + body[i * chunk:(i + 1) * chunk]
        for i in range(count)
    ]


@dataclass
class Message:
    msg_type: int
    msg_id: int
    body: bytes
    first_arrival: float
    complete: float
    fragments: int
    wire_bytes: int  # datagram bytes received, headers included


@dataclass
class _Partial:
    count: int
    first_arrival: float
    parts: dict[int, bytes] = field(default_factory=dict)
    wire_bytes: int = 0


@dataclass
class Incomplete:
    msg_type: int
    msg_id: int
    received: int
    expected: int
    first_arrival: float


class Reassembler:
    """Collect fragments into messages; expire messages that never complete."""

    def __init__(self) -> None:
        self._partial: dict[tuple[int, int], _Partial] = {}
        self.malformed = 0

    def add(self, datagram: bytes, now: float) -> Message | None:
        if len(datagram) < FRAGMENT_HEADER.size:
            self.malformed += 1
            return None
        magic, version, msg_type, msg_id, index, count = FRAGMENT_HEADER.unpack_from(datagram)
        if magic != MAGIC or version != VERSION or index >= count:
            self.malformed += 1
            return None
        key = (msg_type, msg_id)
        partial = self._partial.get(key)
        if partial is None:
            partial = self._partial[key] = _Partial(count, now)
        if index in partial.parts:
            return None  # duplicate
        partial.parts[index] = datagram[FRAGMENT_HEADER.size:]
        partial.wire_bytes += len(datagram)
        if len(partial.parts) < partial.count:
            return None
        del self._partial[key]
        body = b"".join(partial.parts[i] for i in range(partial.count))
        return Message(msg_type, msg_id, body, partial.first_arrival, now, partial.count, partial.wire_bytes)

    def expire(self, older_than: float) -> list[Incomplete]:
        """Drop partial messages whose first fragment arrived before ``older_than``."""

        stale = [k for k, p in self._partial.items() if p.first_arrival < older_than]
        out = []
        for key in stale:
            p = self._partial.pop(key)
            out.append(Incomplete(key[0], key[1], len(p.parts), p.count, p.first_arrival))
        return out


# -- payloads -----------------------------------------------------------------

def encode_image(image: Image.Image, codec: str, jpeg_quality: int = 90) -> bytes:
    """Codec byte + image. ``raw`` and ``png`` are lossless; ``jpeg`` is not."""

    image = image if image.mode == "RGB" else image.convert("RGB")
    if codec == "raw":
        return _RAW_SIZE.pack(*image.size) + image.tobytes()
    buf = io.BytesIO()
    if codec == "png":
        image.save(buf, format="PNG", compress_level=1)
    elif codec == "jpeg":
        image.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        raise ValueError(f"unknown codec {codec!r}; choose from {sorted(CODECS)}")
    return buf.getvalue()


def decode_image(codec_id: int, data: bytes) -> Image.Image:
    if _CODEC_NAMES.get(codec_id) == "raw":
        width, height = _RAW_SIZE.unpack_from(data)
        return Image.frombytes("RGB", (width, height), data[_RAW_SIZE.size:])
    if codec_id not in _CODEC_NAMES:
        raise ValueError(f"unknown codec id {codec_id}")
    return Image.open(io.BytesIO(data)).convert("RGB")


@dataclass(frozen=True)
class DetectRequest:
    frame_id: int
    send_s: float
    conf: float
    iou: float
    imgsz: int
    codec: int
    image_data: bytes


def pack_request(req: DetectRequest) -> bytes:
    return REQUEST_META.pack(req.frame_id, req.send_s, req.conf, req.iou, req.imgsz, req.codec) + req.image_data


def unpack_request(body: bytes) -> DetectRequest:
    frame_id, send_s, conf, iou, imgsz, codec = REQUEST_META.unpack_from(body)
    return DetectRequest(frame_id, send_s, conf, iou, imgsz, codec, body[REQUEST_META.size:])


@dataclass(frozen=True)
class DetectResponse:
    frame_id: int
    request_send_s: float
    request_arrival_s: float  # ground clock == edge clock (one host)
    compute_ms: float  # ground: decode + detect
    response_send_s: float
    detections: list[tuple[np.ndarray, float]]


def pack_response(resp: DetectResponse) -> bytes:
    rows = np.array([[*box, conf] for box, conf in resp.detections], dtype="<f8").reshape(-1, 5)
    meta = RESPONSE_META.pack(
        resp.frame_id, resp.request_send_s, resp.request_arrival_s, resp.compute_ms,
        resp.response_send_s, len(resp.detections),
    )
    return meta + rows.tobytes()


def unpack_response(body: bytes) -> DetectResponse:
    frame_id, req_send, req_arrival, compute_ms, resp_send, n = RESPONSE_META.unpack_from(body)
    rows = np.frombuffer(body, dtype="<f8", count=n * 5, offset=RESPONSE_META.size).reshape(n, 5)
    detections = [(rows[i, :4].astype(np.float64), float(rows[i, 4])) for i in range(n)]
    return DetectResponse(frame_id, req_send, req_arrival, compute_ms, resp_send, detections)


def pack_info(info: dict) -> bytes:
    return json.dumps(info).encode("utf-8")


def unpack_info(body: bytes) -> dict:
    return json.loads(body.decode("utf-8")) if body else {}
