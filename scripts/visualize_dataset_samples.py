"""Render annotation and SSI crop verification images from unified data."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.data.crops import (
    CropConfig,
    bbox_image_to_ssi,
    crop_ssi_with_targets,
    xywh_to_xyxy,
)
from jmdst.data.io import iter_sequence_dirs, read_sequence, resolve_image_path
from jmdst.data.schema import FrameRecord, ObjectAnnotation, SequenceInfo


COLORS = {
    "car": (0, 200, 80),
    "van": (255, 170, 0),
    "truck": (40, 140, 255),
    "bus": (225, 70, 70),
    "selected": (255, 255, 0),
    "anchor": (255, 255, 255),
}


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text)
    pad = 2
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=color)


def _scale_panel(image: Image.Image, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize(size, resampling), scale


def _draw_full_frame(
    image: Image.Image,
    record: FrameRecord,
    selected: ObjectAnnotation,
) -> Image.Image:
    panel, scale = _scale_panel(image, max_width=640, max_height=420)
    draw = ImageDraw.Draw(panel)
    for obj in record.objects:
        color = COLORS["selected"] if obj.track_id == selected.track_id else COLORS.get(obj.class_name, (0, 200, 80))
        x1, y1, x2, y2 = xywh_to_xyxy(obj.bbox_xywh)
        box = tuple(int(round(v * scale)) for v in (x1, y1, x2, y2))
        draw.rectangle(box, outline=color, width=2)
        _draw_label(
            draw,
            (box[0] + 2, max(0, box[1] - 13)),
            f"id{obj.track_id} {obj.class_name}",
            color,
        )
    return panel


def _draw_crop_panel(
    image: Image.Image,
    record: FrameRecord,
    selected: ObjectAnnotation,
    crop_config: CropConfig,
    require_positive_anchor: bool,
    rng: random.Random,
) -> Image.Image:
    ssi, window, targets = crop_ssi_with_targets(
        image,
        selected,
        record.objects,
        config=crop_config,
        rng=rng,
        require_positive_anchor=require_positive_anchor,
    )
    scale = 4
    resampling = getattr(Image, "Resampling", Image).NEAREST
    panel = ssi.resize((crop_config.output_size * scale, crop_config.output_size * scale), resampling)
    draw = ImageDraw.Draw(panel)

    cell = crop_config.output_size * scale // 2
    draw.line((cell, 0, cell, panel.height), fill=COLORS["anchor"], width=1)
    draw.line((0, cell, panel.width, cell), fill=COLORS["anchor"], width=1)

    for obj in record.objects:
        bbox = bbox_image_to_ssi(obj.bbox_xywh, window)
        x1, y1, x2, y2 = xywh_to_xyxy(bbox)
        box = tuple(int(round(v * scale)) for v in (x1, y1, x2, y2))
        color = COLORS["selected"] if obj.track_id == selected.track_id else COLORS.get(obj.class_name, (0, 200, 80))
        draw.rectangle(box, outline=color, width=2)
        _draw_label(draw, (max(0, box[0] + 2), max(0, box[1] + 2)), f"id{obj.track_id}", color)

    for idx, confidence in enumerate(targets["confidences"].tolist()):
        ax, ay, aw, ah = targets["anchors_xywh"][idx]
        x = int((ax + 2) * scale)
        y = int((ay + 2) * scale)
        label = f"a{idx}:c{int(confidence)}"
        _draw_label(draw, (x, y), label, COLORS["anchor"])
    return panel


def collect_samples(
    unified_root: Path,
    dataset: str | None,
    split: str | None,
) -> list[tuple[Path, SequenceInfo, FrameRecord, ObjectAnnotation]]:
    samples = []
    for sequence_dir in iter_sequence_dirs(unified_root, dataset=dataset, split=split):
        info, records = read_sequence(sequence_dir)
        for record in records:
            for obj in record.objects:
                samples.append((sequence_dir, info, record, obj))
    return samples


def render_sample(
    output_path: Path,
    sequence_dir: Path,
    info: SequenceInfo,
    record: FrameRecord,
    selected: ObjectAnnotation,
    crop_config: CropConfig,
    require_positive_anchor: bool,
    rng: random.Random,
) -> None:
    image_path = resolve_image_path(sequence_dir, record.image_path)
    image = Image.open(image_path).convert("RGB")
    full_panel = _draw_full_frame(image, record, selected)
    crop_panel = _draw_crop_panel(
        image,
        record,
        selected,
        crop_config,
        require_positive_anchor=require_positive_anchor,
        rng=rng,
    )

    width = full_panel.width + crop_panel.width + 24
    height = max(full_panel.height, crop_panel.height) + 44
    canvas = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    title = f"{info.dataset}/{info.split}/{info.sequence} frame {record.frame_id} track {selected.track_id}"
    draw.text((12, 10), title, fill=(240, 240, 240))
    canvas.paste(full_panel, (12, 34))
    canvas.paste(crop_panel, (full_panel.width + 24, 34))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--asymmetric-policy", default="paper_min", choices=["paper_min", "base", "ensure_target"])
    parser.add_argument("--random-shift-px", type=float, default=16.0)
    parser.add_argument(
        "--allow-zero-positive-anchor",
        action="store_true",
        help="Do not retry shifted crops when the selected target owns no positive anchor.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = collect_samples(Path(args.unified_root), dataset=args.dataset, split=args.split)
    if not samples:
        raise RuntimeError("No annotated samples found in unified dataset.")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    selected_samples = samples[: args.num_samples]
    crop_config = CropConfig(
        random_shift_px=args.random_shift_px,
        asymmetric_policy=args.asymmetric_policy,
    )
    crop_rng = random.Random(args.seed + 1)
    output_dir = Path(args.output_dir)
    for idx, sample in enumerate(selected_samples):
        output_path = output_dir / f"sample_{idx:03d}.png"
        render_sample(
            output_path,
            *sample,
            crop_config=crop_config,
            require_positive_anchor=not args.allow_zero_positive_anchor,
            rng=crop_rng,
        )
        print(output_path)


if __name__ == "__main__":
    main()
