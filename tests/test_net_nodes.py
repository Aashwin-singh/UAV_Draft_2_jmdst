from __future__ import annotations

import io
import random
import threading
import unittest

import numpy as np
from PIL import Image, ImageDraw

from jmdst.net import GroundServer, LinkConfig, LinkRelay, RemoteDetector
from jmdst.net.protocol import (
    DETECT_REQUEST, DetectRequest, DetectResponse, Reassembler, decode_image, encode_image, fragment,
    pack_request, pack_response, unpack_request, unpack_response, CODECS,
)


def _image(size=(160, 90)) -> Image.Image:
    img = Image.new("RGB", size, (30, 40, 50))
    draw = ImageDraw.Draw(img)
    for i in range(0, size[0], 7):
        draw.line((i, 0, size[0] - i, size[1]), fill=(i % 255, 200, 90))
    return img


class FragmentationTests(unittest.TestCase):
    def test_round_trip_out_of_order_with_duplicates(self) -> None:
        body = bytes(random.Random(0).randrange(256) for _ in range(10_000))
        frags = fragment(DETECT_REQUEST, 7, body, mtu=500)
        self.assertTrue(all(len(f) <= 500 for f in frags))
        shuffled = frags + frags[:3]
        random.Random(1).shuffle(shuffled)
        r = Reassembler()
        done = [m for m in (r.add(f, 1.0) for f in shuffled) if m is not None]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0].body, body)
        self.assertEqual(done[0].fragments, len(frags))

    def test_missing_fragment_never_completes_and_expires(self) -> None:
        frags = fragment(DETECT_REQUEST, 9, bytes(5000), mtu=1000)
        r = Reassembler()
        for f in frags[:-1]:
            self.assertIsNone(r.add(f, 1.0))
        self.assertEqual(r.expire(older_than=0.5), [])
        gone = r.expire(older_than=2.0)
        self.assertEqual((gone[0].msg_id, gone[0].received, gone[0].expected), (9, len(frags) - 1, len(frags)))

    def test_garbage_is_ignored(self) -> None:
        r = Reassembler()
        self.assertIsNone(r.add(b"nope", 0.0))
        self.assertIsNone(r.add(b"XX" + bytes(30), 0.0))
        self.assertEqual(r.malformed, 2)


class PayloadTests(unittest.TestCase):
    def test_lossless_codecs_are_pixel_exact(self) -> None:
        img = _image()
        for codec in ("raw", "png"):
            out = decode_image(CODECS[codec], encode_image(img, codec))
            self.assertEqual(out.tobytes(), img.tobytes(), codec)

    def test_jpeg_is_smaller_and_close(self) -> None:
        # Smooth content with a few solid shapes, like aerial road footage --
        # _image()'s 1-px diagonal lines are JPEG's worst case.
        x = np.linspace(0, 255, 320, dtype=np.float64)
        arr = np.stack([np.tile(x, (180, 1)), np.tile(x[::-1], (180, 1)), np.full((180, 320), 90.0)], axis=2)
        img = Image.fromarray(arr.astype(np.uint8))
        ImageDraw.Draw(img).rectangle((100, 60, 140, 80), fill=(250, 250, 250))
        data = encode_image(img, "jpeg", jpeg_quality=90)
        out = decode_image(CODECS["jpeg"], data)
        self.assertEqual(out.size, img.size)
        self.assertLess(len(data), len(encode_image(img, "raw")))
        err = np.abs(np.asarray(out, dtype=int) - np.asarray(img, dtype=int)).mean()
        self.assertLess(err, 10)

    def test_request_and_response_round_trip_bit_exact(self) -> None:
        req = DetectRequest(42, 123.456789, 0.4, 0.2, 640, CODECS["jpeg"], b"\x01\x02")
        self.assertEqual(unpack_request(pack_request(req)), req)
        dets = [(np.array([1.1, 2.2, 3.3, 4.4]), 0.987654321), (np.array([0.1, 0.2, 0.3, 0.4]), 1 / 3)]
        resp = unpack_response(pack_response(DetectResponse(42, 1.0, 2.0, 3.5, 4.0, dets)))
        self.assertEqual(resp.frame_id, 42)
        for (b0, c0), (b1, c1) in zip(dets, resp.detections):
            self.assertTrue(np.array_equal(b0, b1))
            self.assertEqual(c0, c1)
        self.assertEqual(unpack_response(pack_response(DetectResponse(1, 0, 0, 0, 0, []))).detections, [])


class DetectionCacheTests(unittest.TestCase):
    def test_save_load_round_trip_and_parameter_guard(self) -> None:
        import tempfile
        from pathlib import Path

        from jmdst.net import DetectionCache

        cache = DetectionCache({"conf": 0.4, "iou": 0.2, "imgsz": 640, "note": "x"})
        dets = [(np.array([1.0, 2.0, 3.0, 4.0]), 0.5), (np.array([5.0, 6.0, 7.0, 8.0]), 1 / 3)]
        cache.add(b"frame-a", dets, 12.5)
        cache.add(b"frame-b", [], 9.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.npz"
            cache.save(path)
            loaded = DetectionCache.load(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded.meta["note"], "x")
        hit = loaded.lookup(b"frame-a", 0.4, 0.2, 640)
        self.assertEqual(hit.compute_ms, 12.5)
        for (b0, c0), (b1, c1) in zip(dets, hit.detections):
            self.assertTrue(np.array_equal(b0, b1))
            self.assertEqual(c0, c1)
        self.assertEqual(loaded.lookup(b"frame-b", 0.4, 0.2, 640).detections, [])
        self.assertIsNone(loaded.lookup(b"frame-c", 0.4, 0.2, 640))
        self.assertIsNone(loaded.lookup(b"frame-a", 0.55, 0.2, 640))  # other post-processing: miss, not a wrong answer


class _FakeYolo:
    """Deterministic stand-in: one box whose values encode what it was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, image, conf, iou, imgsz):
        self.calls.append((image.size, conf, iou, imgsz))
        w, h = image.size
        return [(np.array([w / 3, h / 7, 10.0, 5.0]), conf + 0.123456789)]


class NodeTests(unittest.TestCase):
    def _ground(self) -> tuple[GroundServer, _FakeYolo, io.StringIO]:
        fake = _FakeYolo()
        log = io.StringIO()
        ground = GroundServer(("127.0.0.1", 0), fake, info={"model": "fake"}, telemetry=log)
        stop = threading.Event()
        thread = threading.Thread(target=ground.serve, args=(stop,), daemon=True)
        thread.start()
        self.addCleanup(lambda: (stop.set(), thread.join(2), ground.close()))
        return ground, fake, log

    def _relay(self, ground_addr, e2g: LinkConfig, g2e: LinkConfig) -> LinkRelay:
        relay = LinkRelay(("127.0.0.1", 0), ground_addr, e2g, g2e)
        stop = threading.Event()
        thread = threading.Thread(target=relay.run, args=(stop,), daemon=True)
        thread.start()
        self.addCleanup(lambda: (stop.set(), thread.join(2), relay.close()))
        return relay

    def test_handshake_and_exact_detections_direct(self) -> None:
        ground, fake, log = self._ground()
        remote = RemoteDetector(ground.address, conf=0.4, iou=0.2, imgsz=640, codec="raw", deadline_s=2.0)
        self.addCleanup(remote.close)
        self.assertEqual(remote.handshake(timeout_s=5)["model"], "fake")

        remote.context = {"sequence": "S", "frame_id": 12}
        img = _image((640, 360))
        dets = remote(img)
        box, conf = dets[0]
        self.assertTrue(np.array_equal(box, np.array([640 / 3, 360 / 7, 10.0, 5.0])))
        self.assertEqual(conf, 0.4 + 0.123456789)
        self.assertEqual(fake.calls, [((640, 360), 0.4, 0.2, 640)])  # edge's settings reach the ground

        ex = remote.exchanges[0]
        self.assertEqual((ex.outcome, ex.context["frame_id"], ex.detections), ("ok", 12, 1))
        self.assertLessEqual(ex.send_s, ex.ground_arrival_s)
        self.assertLessEqual(ex.ground_arrival_s, ex.response_send_s)
        self.assertLessEqual(ex.response_send_s, ex.response_arrival_s)
        self.assertGreater(ex.request_packets, 400)  # 640x360 raw ~ 691 KB in 1400-byte datagrams

    def test_lost_request_returns_none_and_tracker_can_fall_back(self) -> None:
        ground, fake, _ = self._ground()
        relay = self._relay(ground.address, LinkConfig(loss_pct=100.0), LinkConfig())
        remote = RemoteDetector(relay.edge_address, conf=0.5, iou=0.2, imgsz=640, codec="jpeg", deadline_s=0.15)
        self.addCleanup(remote.close)
        self.assertIsNone(remote(_image()))
        self.assertEqual(remote.exchanges[0].outcome, "lost")
        self.assertEqual(fake.calls, [])

    def test_reply_after_deadline_is_discarded_and_marked_late(self) -> None:
        ground, _, _ = self._ground()
        relay = self._relay(ground.address, LinkConfig(delay_ms=120.0), LinkConfig())
        remote = RemoteDetector(relay.edge_address, conf=0.5, iou=0.2, imgsz=640, codec="jpeg", deadline_s=0.05)
        self.addCleanup(remote.close)
        self.assertIsNone(remote(_image()))
        self.assertEqual(remote.exchanges[0].outcome, "lost")
        remote.drain(wait_s=0.5)
        ex = remote.exchanges[0]
        self.assertEqual(ex.outcome, "late")
        self.assertGreaterEqual(ex.uplink_ms, 120.0)

    def test_modelled_ground_replies_from_cache_after_its_compute_time(self) -> None:
        from jmdst.net import DetectionCache

        img = _image((320, 180))
        cache = DetectionCache({"conf": 0.4, "iou": 0.2, "imgsz": 640})
        cached = [(np.array([1.5, 2.5, 30.0, 20.0]), 0.77)]
        cache.add(encode_image(img, "jpeg", 90), cached, compute_ms=60.0)
        fake = _FakeYolo()
        ground = GroundServer(("127.0.0.1", 0), fake, cache=cache)
        stop = threading.Event()
        thread = threading.Thread(target=ground.serve, args=(stop,), daemon=True)
        thread.start()
        self.addCleanup(lambda: (stop.set(), thread.join(2), ground.close()))
        remote = RemoteDetector(ground.address, conf=0.4, iou=0.2, imgsz=640, codec="jpeg", deadline_s=2.0)
        self.addCleanup(remote.close)

        box, conf = remote(img)[0]
        self.assertTrue(np.array_equal(box, cached[0][0]))
        self.assertEqual(conf, 0.77)
        self.assertEqual(fake.calls, [])  # no live YOLO for a hit
        ex = remote.exchanges[0]
        self.assertGreaterEqual(ex.response_send_s - ex.ground_arrival_s, 0.060)  # modelled compute honoured

        remote(_image((160, 90)))  # not in the cache -> live
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual((ground.stats.cache_hits, ground.stats.cache_misses), (1, 1))

    def test_handshake_fails_loudly_when_ground_is_down(self) -> None:
        relay = self._relay(("127.0.0.1", 9), LinkConfig(), LinkConfig())  # nothing listens there
        remote = RemoteDetector(relay.edge_address, conf=0.5, iou=0.2, imgsz=640)
        self.addCleanup(remote.close)
        with self.assertRaises(TimeoutError):
            remote.handshake(timeout_s=0.6, retry_s=0.2)


if __name__ == "__main__":
    unittest.main()
