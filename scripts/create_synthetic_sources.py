"""Create tiny VisDrone/UAVDT-like source folders for pipeline verification."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def _draw_frame(path: Path, boxes: list[tuple[int, int, int, int]], color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (320, 180), (48, 56, 64))
    draw = ImageDraw.Draw(image)
    for idx, (x, y, w, h) in enumerate(boxes):
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        draw.text((x + 3, y + 3), str(idx + 1), fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def create_visdrone(root: Path) -> None:
    sequence = "uav000001"
    seq_dir = root / "visdrone_src" / "VisDrone2019-MOT-train" / "sequences" / sequence
    ann_dir = root / "visdrone_src" / "VisDrone2019-MOT-train" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for frame in range(1, 4):
        car = (40 + frame * 8, 50, 42, 24)
        van = (160, 65 + frame * 4, 50, 28)
        _draw_frame(seq_dir / f"{frame:07d}.jpg", [car, van], (0, 220, 120))
        lines.append(f"{frame},1,{car[0]},{car[1]},{car[2]},{car[3]},1,4,0,0\n")
        lines.append(f"{frame},2,{van[0]},{van[1]},{van[2]},{van[3]},1,5,0,1\n")
    (ann_dir / f"{sequence}.txt").write_text("".join(lines), encoding="utf-8")


def create_uavdt(root: Path) -> None:
    sequence = "M0101"
    seq_dir = root / "uavdt_src" / "UAV-benchmark-M" / sequence
    gt_dir = root / "uavdt_src" / "GT"
    gt_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for frame in range(1, 4):
        car = (70 + frame * 10, 90, 44, 22)
        truck = (210, 40 + frame * 5, 56, 30)
        _draw_frame(seq_dir / f"img{frame:06d}.jpg", [car, truck], (70, 150, 255))
        lines.append(f"{frame},1,{car[0]},{car[1]},{car[2]},{car[3]},0,0,1\n")
        lines.append(f"{frame},2,{truck[0]},{truck[1]},{truck[2]},{truck[3]},0,1,2\n")
    (gt_dir / f"{sequence}_gt_whole.txt").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    create_visdrone(root)
    create_uavdt(root)
    print(root / "visdrone_src")
    print(root / "uavdt_src")


if __name__ == "__main__":
    main()
