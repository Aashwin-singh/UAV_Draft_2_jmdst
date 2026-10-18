from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from jmdst.sim import CameraFeed, EdgeHardware, format_cpu_list, latest_finished, parse_cpu_list


class StreamingScoringTests(unittest.TestCase):
    def test_frame_shows_newest_result_finished_by_its_capture(self) -> None:
        # Results finished at 0.05, 0.12, 0.13; frames captured every 0.04 s.
        captures = [0.00, 0.04, 0.08, 0.12, 0.16]
        self.assertEqual(latest_finished([0.05, 0.12, 0.13], captures), [-1, -1, 0, 1, 2])

    def test_no_results_shows_nothing(self) -> None:
        self.assertEqual(latest_finished([], [0.0, 1.0]), [-1, -1])


class HardwareSpecTests(unittest.TestCase):
    def test_cpu_list_round_trip(self) -> None:
        self.assertEqual(parse_cpu_list("16-19,24"), (16, 17, 18, 19, 24))
        self.assertEqual(format_cpu_list((16, 17, 18, 19, 24)), "16-19,24")
        self.assertEqual(format_cpu_list((3,)), "3")
        self.assertIsNone(parse_cpu_list(None))
        with self.assertRaises(ValueError):
            parse_cpu_list("0-9999")

    def test_thread_count_fits_the_tightest_limit(self) -> None:
        self.assertEqual(EdgeHardware(cpus=(16, 17, 18, 19)).thread_count(), 4)
        self.assertEqual(EdgeHardware(cpu_cores=1.5).thread_count(), 2)
        self.assertEqual(EdgeHardware(cpu_cores=1.5, cpus=(16,)).thread_count(), 1)
        self.assertEqual(EdgeHardware(threads=3, cpu_cores=1.0).thread_count(), 3)
        self.assertIsNone(EdgeHardware().thread_count())
        self.assertFalse(EdgeHardware().constrained)

    @unittest.skipUnless(sys.platform == "win32", "Job Objects are Windows-only")
    def test_cpu_cap_holds_in_a_child_process(self) -> None:
        # Four GIL-free busy threads would use ~4 cores; capped at 0.5 they must
        # not get far past it. Upper bound only, so background load cannot
        # make this flaky.
        code = (
            "import hashlib,threading,time,psutil\n"
            "from jmdst.sim import EdgeHardware, apply_edge_hardware\n"
            "apply_edge_hardware(EdgeHardware(cpu_cores=0.5))\n"
            "buf=bytes(32<<20); p=psutil.Process(); c0=sum(p.cpu_times()[:2]); t0=time.perf_counter()\n"
            "def w():\n"
            "    while time.perf_counter()-t0<1.5: hashlib.sha256(buf).digest()\n"
            "ts=[threading.Thread(target=w) for _ in range(4)]; [t.start() for t in ts]; [t.join() for t in ts]\n"
            "print((sum(p.cpu_times()[:2])-c0)/(time.perf_counter()-t0))\n"
        )
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertLess(float(out.stdout.strip().splitlines()[-1]), 0.5 * 1.4)


class CameraFeedTests(unittest.TestCase):
    def _frames(self, tmp: Path, n: int) -> list[Path]:
        paths = []
        for k in range(n):
            path = tmp / f"{k:04d}.png"
            Image.new("RGB", (48, 32), (k * 9 % 256, 100, 200)).save(path)
            paths.append(path)
        return paths

    def test_fast_consumer_sees_frames_in_order_with_correct_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._frames(Path(tmp), 15)
            feed = CameraFeed(paths, (48, 32), fps=60)
            try:
                feed.start()
                taken, last = [], -1
                while (got := feed.take_newest(last)) is not None:
                    k, image = got
                    self.assertGreater(k, last)
                    self.assertEqual(image.getpixel((5, 5)), (k * 9 % 256, 100, 200))
                    self.assertGreaterEqual(time.perf_counter(), feed.capture_time(k))  # never before capture
                    taken.append(k)
                    last = k
                self.assertEqual(taken[-1], 14)
                self.assertGreaterEqual(len(taken), 12)  # a fast consumer misses (almost) nothing
                self.assertLess(feed.max_lateness_ms, 30.0)
            finally:
                feed.close()

    def test_video_source(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            for k in range(12):
                writer.write(np.full((48, 64, 3), (40, 120, k * 20), dtype=np.uint8))  # BGR
            writer.release()
            feed = CameraFeed(path, (64, 48), fps=60, n=12)
            try:
                feed.start()
                last, taken = -1, []
                while (got := feed.take_newest(last)) is not None:
                    k, image = got
                    r, g, b = image.getpixel((32, 24))
                    self.assertLess(abs(r - k * 20) + abs(g - 120) + abs(b - 40), 30)  # BGR -> RGB, lossy codec
                    taken.append(k)
                    last = k
                self.assertEqual(taken[-1], 11)
            finally:
                feed.close()

    def test_slow_consumer_skips_to_the_newest_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._frames(Path(tmp), 20)
            feed = CameraFeed(paths, (48, 32), fps=100)  # a frame every 10 ms
            try:
                feed.start()
                taken, last = [], -1
                while (got := feed.take_newest(last)) is not None:
                    taken.append(got[0])
                    last = got[0]
                    time.sleep(0.035)  # ~3.5 frame periods of "processing"
                gaps = [b - a for a, b in zip(taken, taken[1:])]
                self.assertLessEqual(len(taken), 10)
                self.assertGreaterEqual(min(gaps), 2)  # frames were skipped, never repeated
            finally:
                feed.close()


if __name__ == "__main__":
    unittest.main()
