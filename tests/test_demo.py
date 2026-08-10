"""Tests for the demo renderer and frame-source handling (scripts/demo.py).

Covers the pure helpers only -- running the full pipeline needs trained
checkpoints, which are gitignored.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.demo import draw_hud, draw_outputs, open_source, track_color


class _Out:
    """Stand-in for pipeline.ConfirmedOutput."""

    def __init__(self, track_id: int, bbox, confidence: float = 0.9) -> None:
        self.track_id = track_id
        self.bbox_xywh = np.asarray(bbox, dtype=np.float64)
        self.confidence = confidence


class TrackColorTest(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        self.assertEqual(track_color(7), track_color(7))

    def test_distinct_for_consecutive_ids(self) -> None:
        colours = {track_color(i) for i in range(1, 20)}
        self.assertEqual(len(colours), 19)

    def test_returns_valid_bgr(self) -> None:
        for channel in track_color(123):
            self.assertIsInstance(channel, int)
            self.assertTrue(0 <= channel <= 255)


class DrawOutputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_draws_something(self) -> None:
        draw_outputs(self.frame, [_Out(1, [50.0, 60.0, 40.0, 30.0])])
        self.assertGreater(int(self.frame.sum()), 0)

    def test_no_outputs_leaves_frame_untouched(self) -> None:
        draw_outputs(self.frame, [])
        self.assertEqual(int(self.frame.sum()), 0)

    def test_box_at_top_edge_keeps_label_in_frame(self) -> None:
        # Label is drawn above the box; at y=0 it must not be clipped away.
        draw_outputs(self.frame, [_Out(42, [10.0, 0.0, 40.0, 30.0])])
        self.assertGreater(int(self.frame[0:20, 0:80].sum()), 0)

    def test_survives_box_partly_outside_the_frame(self) -> None:
        draw_outputs(self.frame, [_Out(3, [-20.0, -15.0, 60.0, 50.0])])
        draw_outputs(self.frame, [_Out(4, [300.0, 220.0, 80.0, 60.0])])

    def test_float_boxes_are_accepted(self) -> None:
        draw_outputs(self.frame, [_Out(5, [10.7, 20.2, 30.9, 15.4])])
        self.assertGreater(int(self.frame.sum()), 0)


class DrawHudTest(unittest.TestCase):
    def test_writes_into_the_top_left(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_hud(frame, ["frame 1", "DETECT", "tracks: 4"])
        self.assertGreater(int(frame[0:80, 0:200].sum()), 0)
        # Lower-right region must be untouched by the overlay.
        self.assertEqual(int(frame[200:240, 250:320].sum()), 0)


class OpenSourceTest(unittest.TestCase):
    def test_reads_a_directory_of_images_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                img = np.full((32, 32, 3), i * 40, dtype=np.uint8)
                cv2.imwrite(str(root / f"{i:04d}.jpg"), img)

            frames, fps, total = open_source(root, None, 25.0)
            collected = list(frames)

        self.assertEqual(total, 3)
        self.assertEqual(fps, 25.0)
        self.assertEqual(len(collected), 3)
        # Sorted order means increasing brightness.
        means = [float(f.mean()) for f in collected]
        self.assertEqual(means, sorted(means))

    def test_max_frames_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(5):
                cv2.imwrite(str(root / f"{i:04d}.png"), np.zeros((16, 16, 3), dtype=np.uint8))
            frames, _fps, total = open_source(root, 2, 30.0)
            self.assertEqual(total, 2)
            self.assertEqual(len(list(frames)), 2)

    def test_missing_source_raises(self) -> None:
        with self.assertRaises(SystemExit):
            open_source(Path("definitely-not-here"), None, 30.0)

    def test_directory_without_images_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                open_source(Path(tmp), None, 30.0)

    def test_unsupported_file_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("x", encoding="utf-8")
            with self.assertRaises(SystemExit):
                open_source(path, None, 30.0)


if __name__ == "__main__":
    unittest.main()
