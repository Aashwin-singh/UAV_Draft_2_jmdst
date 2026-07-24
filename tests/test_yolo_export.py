from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from test_data_pipeline import _make_uavdt_source, _make_visdrone_source

from jmdst.data.converters import convert_uavdt, convert_visdrone
from jmdst.data.yolo_export import export_yolo_dataset


class YoloExportTests(unittest.TestCase):
    def test_export_creates_expected_layout_and_normalized_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unified = root / "unified"

            visdrone_source = _make_visdrone_source(root)
            convert_visdrone(visdrone_source, unified, split="train")

            uavdt_source = _make_uavdt_source(root)
            convert_uavdt(uavdt_source, unified, split="val")

            yolo_root = root / "yolo"
            stats = export_yolo_dataset(
                unified_root=unified,
                output_root=yolo_root,
                datasets=["visdrone", "uavdt"],
                splits=["train", "val", "test"],
            )

            # visdrone: 1 sequence x 2 frames = 2 images (train)
            # uavdt: 1 sequence x 2 frames = 2 images (val)
            self.assertEqual(stats.images_written, 4)
            self.assertEqual(stats.labels_written, 4)
            self.assertEqual(stats.per_split_images.get("train"), 2)
            self.assertEqual(stats.per_split_images.get("val"), 2)
            self.assertNotIn("test", stats.per_split_images)

            train_images = sorted((yolo_root / "images" / "train").iterdir())
            train_labels = sorted((yolo_root / "labels" / "train").iterdir())
            self.assertEqual(len(train_images), 2)
            self.assertEqual(len(train_labels), 2)
            self.assertEqual(
                {p.stem for p in train_images},
                {p.stem for p in train_labels},
            )

            # visdrone synthetic source: car (4) and bus (9) source classes only
            # (the third annotation, source class 1 "pedestrian", is dropped by the converter).
            label_text = train_labels[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(label_text), 2)
            for line in label_text:
                parts = line.split()
                self.assertEqual(len(parts), 5)
                class_id = int(parts[0])
                x_center, y_center, width, height = (float(v) for v in parts[1:])
                self.assertIn(class_id, {0, 1, 2, 3})
                for value in (x_center, y_center, width, height):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

            # test hardlink-or-copy actually placed real, non-empty image bytes
            for image_path in train_images:
                self.assertGreater(image_path.stat().st_size, 0)

            yaml_path = yolo_root / "dataset.yaml"
            self.assertTrue(yaml_path.is_file())
            yaml_text = yaml_path.read_text(encoding="utf-8")
            self.assertIn("train: images/train", yaml_text)
            self.assertIn("val: images/val", yaml_text)
            self.assertNotIn("test: images/test", yaml_text)


if __name__ == "__main__":
    unittest.main()
