from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from test_data_pipeline import _make_visdrone_source

from jmdst.data.converters import convert_visdrone
from scripts.verify_converted_dataset import verify_unified_dataset


class VerifyConvertedDatasetTests(unittest.TestCase):
    def test_synthetic_unified_dataset_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_visdrone_source(root)
            unified = root / "unified"
            convert_visdrone(source, unified, split="train")

            stats = verify_unified_dataset(unified, dataset="visdrone", split="train", verify_ssi=True)

            self.assertTrue(stats.ready_for_yolov11)
            self.assertEqual(stats.sequence_dirs, 1)
            self.assertEqual(stats.valid_sequences, 1)
            self.assertEqual(stats.frames, 2)
            self.assertEqual(stats.annotations, 4)
            self.assertEqual(stats.ssi_crops_generated, 4)
            self.assertFalse(stats.invalid_ssi_crops)


if __name__ == "__main__":
    unittest.main()
