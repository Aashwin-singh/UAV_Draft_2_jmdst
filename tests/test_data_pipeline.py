from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from jmdst.data.converters import convert_uavdt, convert_visdrone
from jmdst.data.crops import (
    CropConfig,
    bbox_image_to_ssi,
    bbox_ssi_to_image,
    crop_ssi,
    overlap_to_bbox,
    overlap_vector,
    paper_search_side,
)
from jmdst.data.datasets import FELNetSSIDataset, UnifiedDetectionDataset
from jmdst.data.io import iter_sequence_dirs, read_sequence


def _draw_image(path: Path, boxes: list[tuple[int, int, int, int]]) -> None:
    image = Image.new("RGB", (320, 180), (40, 50, 60))
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x, y, w, h = box
        draw.rectangle((x, y, x + w, y + h), outline=(0, 230, 120), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _make_visdrone_source(root: Path) -> Path:
    source = root / "visdrone_src"
    sequence = "uav000001"
    seq_dir = source / "VisDrone2019-MOT-train" / "sequences" / sequence
    ann_dir = source / "VisDrone2019-MOT-train" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for frame in range(1, 3):
        car = (40 + frame * 5, 50, 42, 24)
        bus = (150, 70 + frame * 3, 55, 26)
        _draw_image(seq_dir / f"{frame:07d}.jpg", [car, bus])
        lines.append(f"{frame},1,{car[0]},{car[1]},{car[2]},{car[3]},1,4,0,0\n")
        lines.append(f"{frame},2,{bus[0]},{bus[1]},{bus[2]},{bus[3]},1,9,0,1\n")
        lines.append(f"{frame},3,5,5,10,10,1,1,0,0\n")
    (ann_dir / f"{sequence}.txt").write_text("".join(lines), encoding="utf-8")
    return source


def _make_uavdt_source(root: Path) -> Path:
    source = root / "uavdt_src"
    sequence = "M0101"
    seq_dir = source / "UAV-benchmark-M" / sequence
    gt_dir = source / "GT"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for frame in range(1, 3):
        car = (70 + frame * 5, 88, 40, 20)
        truck = (190, 42, 54, 30)
        _draw_image(seq_dir / f"img{frame:06d}.jpg", [car, truck])
        lines.append(f"{frame},1,{car[0]},{car[1]},{car[2]},{car[3]},0,0,1\n")
        lines.append(f"{frame},2,{truck[0]},{truck[1]},{truck[2]},{truck[3]},0,0,2\n")
    (gt_dir / f"{sequence}_gt_whole.txt").write_text("".join(lines), encoding="utf-8")
    # UAVDT ignore-region file: per-frame don't-care rectangles.
    ignore_lines = [f"{frame},1,300,10,50,50,1,-1,-1\n" for frame in range(1, 3)]
    (gt_dir / f"{sequence}_gt_ignore.txt").write_text("".join(ignore_lines), encoding="utf-8")
    return source


class CropTests(unittest.TestCase):
    def test_paper_search_side(self) -> None:
        config = CropConfig(asymmetric_ratio_threshold=999)
        expected = math.sqrt((1.3 * 20 + 0.3 * 10) * (1.3 * 10 + 0.3 * 20))
        self.assertAlmostEqual(paper_search_side(20, 10, config), expected)

    def test_overlap_round_trip(self) -> None:
        anchor = [0, 0, 32, 32]
        bbox = [10, 8, 20, 16]
        overlap = overlap_vector(anchor, bbox)
        restored = overlap_to_bbox(anchor, overlap)
        for actual, expected in zip(restored, bbox):
            self.assertAlmostEqual(actual, expected)

    def test_crop_size_and_coordinate_round_trip(self) -> None:
        image = Image.new("RGB", (100, 80), (20, 20, 20))
        bbox = [30, 20, 20, 10]
        ssi, window = crop_ssi(image, bbox, CropConfig(asymmetric_ratio_threshold=999))
        self.assertEqual(ssi.size, (64, 64))
        ssi_bbox = bbox_image_to_ssi(bbox, window)
        restored = bbox_ssi_to_image(ssi_bbox, window)
        for actual, expected in zip(restored, bbox):
            self.assertAlmostEqual(actual, expected)


class ConverterAndDatasetTests(unittest.TestCase):
    def test_visdrone_converter_and_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_visdrone_source(root)
            output = root / "unified"
            converted = convert_visdrone(source, output, split="train")
            self.assertEqual(len(converted), 1)

            info, records = read_sequence(converted[0])
            self.assertEqual(info.dataset, "visdrone")
            self.assertEqual(len(records), 2)
            self.assertEqual(len(records[0].objects), 2)
            self.assertEqual({obj.class_name for obj in records[0].objects}, {"car", "bus"})

            detection = UnifiedDetectionDataset(output, dataset="visdrone", split="train")
            self.assertEqual(len(detection), 2)
            _, target = detection[0]
            self.assertEqual(target["boxes"].shape[-1], 4)

            felnet = FELNetSSIDataset(output, dataset="visdrone", split="train", training=False)
            self.assertEqual(len(felnet), 4)
            ssi, felnet_target = felnet[0]
            self.assertEqual(tuple(ssi.shape[-2:]), (64, 64))
            self.assertEqual(felnet_target["overlaps"].shape, (4, 4))

            felnet_train = FELNetSSIDataset(
                output,
                dataset="visdrone",
                split="train",
                training=True,
                seed=2,
                require_positive_anchor=True,
            )
            _, train_target = felnet_train[0]
            positive_track_ids = train_target["track_ids"][train_target["confidences"] > 0]
            self.assertIn(train_target["center_track_id"], positive_track_ids.tolist())

    def test_uavdt_converter_car_only_and_all_vehicle_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_uavdt_source(root)
            output = root / "unified"

            converted_car_only = convert_uavdt(source, output, split="train", car_only=True)
            _, records = read_sequence(converted_car_only[0])
            self.assertEqual(len(records[0].objects), 1)
            self.assertEqual(records[0].objects[0].class_name, "car")

            # UAVDT ignore regions are extracted alongside the annotations.
            ignore_path = converted_car_only[0] / "ignore_regions.jsonl"
            self.assertTrue(ignore_path.is_file())
            import json

            first = json.loads(ignore_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["regions_xywh"], [[300.0, 10.0, 50.0, 50.0]])

            output_all = root / "unified_all"
            converted_all = convert_uavdt(source, output_all, split="train", car_only=False)
            _, records_all = read_sequence(converted_all[0])
            self.assertEqual({obj.class_name for obj in records_all[0].objects}, {"car", "truck"})

    def test_sequence_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_visdrone_source(root)
            output = root / "unified"
            convert_visdrone(source, output, split="train")
            sequence_dirs = iter_sequence_dirs(output, dataset="visdrone", split="train")
            self.assertEqual(len(sequence_dirs), 1)


if __name__ == "__main__":
    unittest.main()
