"""Generate the figure set for presenting this reproduction (per-phase evidence).

Produces, into --output-dir:
  fig1_dataset_gt.jpg        GT boxes on a real frame (Phase 1)
  fig2_ssi_crops.jpg         64x64 SSI crops + anchor grid (Phase 1/3)
  fig3_yolo_detections.jpg   YOLO detections at the paper's conf/NMS (Phase 2)
  fig4_yolo_curve.png        YOLO val mAP vs epoch, original vs retrained (Phase 2)
  fig5_felnet_loss.png       FELNet training loss, original vs rebalanced (Phase 4)
  fig6_felnet_boxes.jpg      FELNet predicted vs GT box inside SSIs (Phase 4)
  fig7_embedding_sep.png     same- vs different-identity similarity (Phase 4/11)
  fig8_tracking.jpg          tracked IDs over time, detection vs tracking frames (Phase 9)
  fig9_tau_sweep.png         MOTA/HOTA/FPS vs detection interval tau (Phase 11)

Usage:
    python scripts/make_presentation_figures.py --output-dir outputs/presentation
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.data.crops import CropConfig, anchor_boxes, crop_ssi, crop_ssi_with_targets, overlap_to_bbox
from jmdst.data.io import iter_sequence_dirs, read_sequence, resolve_image_path
from jmdst.models import FELNet, FELNetConfig
from jmdst.pipeline import JMDSTTracker
from jmdst.pipeline.models import FELNetLocalizer, YoloDetector, load_felnet

PALETTE = [
    (66, 135, 245), (245, 130, 32), (60, 190, 120), (230, 80, 90),
    (160, 110, 230), (240, 200, 40), (40, 200, 210), (240, 120, 200),
]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _seq(dataset: str, split: str, name: str | None = None):
    dirs = iter_sequence_dirs("data/unified", dataset=dataset, split=split)
    if name:
        dirs = [d for d in dirs if d.name == name] or dirs
    return dirs[0]


def _busiest_frame(records, limit=None):
    pool = records if limit is None else records[:limit]
    return max(pool, key=lambda r: len(r.objects))


def _save(img: Image.Image, path: Path, max_width=1100):
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)), Image.LANCZOS)
    img.convert("RGB").save(path, quality=88)
    print(f"  wrote {path}")


def fig1_dataset_gt(out: Path) -> None:
    seq = _seq("visdrone", "val", "uav0000077_00720_v")
    info, records = read_sequence(seq)
    record = _busiest_frame(records)
    img = Image.open(resolve_image_path(seq, record.image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, obj in enumerate(record.objects):
        x, y, w, h = obj.bbox_xywh
        draw.rectangle([x, y, x + w, y + h], outline=PALETTE[i % len(PALETTE)], width=3)
    _save(img, out / "fig1_dataset_gt.jpg")


def fig2_ssi_crops(out: Path) -> None:
    seq = _seq("visdrone", "val", "uav0000077_00720_v")
    info, records = read_sequence(seq)
    record = _busiest_frame(records)
    img = Image.open(resolve_image_path(seq, record.image_path)).convert("RGB")

    cell, n = 128, 6
    canvas = Image.new("RGB", (cell * n, cell), (24, 26, 30))
    for i, obj in enumerate(record.objects[:n]):
        ssi, _ = crop_ssi(img, obj.bbox_xywh, config=CropConfig(random_shift_px=0.0))
        tile = ssi.resize((cell, cell), Image.NEAREST)
        d = ImageDraw.Draw(tile)
        # 2x2 anchor grid the FELNet heads predict over
        d.line([(cell // 2, 0), (cell // 2, cell)], fill=(255, 255, 255), width=1)
        d.line([(0, cell // 2), (cell, cell // 2)], fill=(255, 255, 255), width=1)
        canvas.paste(tile, (i * cell, 0))
    _save(canvas, out / "fig2_ssi_crops.jpg", max_width=1100)


def fig3_yolo_detections(out: Path, weights: str) -> None:
    from ultralytics import YOLO

    seq = _seq("visdrone", "val", "uav0000077_00720_v")
    info, records = read_sequence(seq)
    record = _busiest_frame(records)
    img = Image.open(resolve_image_path(seq, record.image_path)).convert("RGB")

    det = YoloDetector(YOLO(weights), conf=0.55, iou=0.2)
    draw = ImageDraw.Draw(img)
    for i, (box, conf) in enumerate(det(img)):
        x, y, w, h = box
        c = PALETTE[i % len(PALETTE)]
        draw.rectangle([x, y, x + w, y + h], outline=c, width=3)
        draw.text((x, max(0, y - 12)), f"{conf:.2f}", fill=c)
    _save(img, out / "fig3_yolo_detections.jpg")


def fig4_yolo_curve(out: Path) -> None:
    runs = {"original (paper defaults)": "full_run", "retrained (+strong aug, early stop)": "retrain_aug"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, run in runs.items():
        path = Path(f"outputs/yolo_runs/{run}/results.csv")
        if not path.is_file():
            continue
        rows = list(csv.DictReader(path.open()))
        ep = [int(r["epoch"]) for r in rows]
        m = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        ax.plot(ep, m, label=label, linewidth=1.8)
        best = int(np.argmax(m))
        ax.scatter([ep[best]], [m[best]], zorder=5, s=45)
        ax.annotate(f"peak ep{ep[best]}\n{m[best]:.3f}", (ep[best], m[best]),
                    textcoords="offset points", xytext=(8, -18), fontsize=8)
    ax.set_xlabel("epoch"); ax.set_ylabel("val mAP50-95")
    ax.set_title("YOLOv11n validation mAP — overfitting fixed by augmentation + early stopping")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig4_yolo_curve.png"); plt.close(fig)
    print(f"  wrote {out / 'fig4_yolo_curve.png'}")


def fig5_felnet_loss(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, run in {"original (pixel overlap targets)": "full_run",
                       "rebalanced (normalized targets)": "rebalanced"}.items():
        path = Path(f"outputs/felnet_runs/{run}/history.json")
        if not path.is_file():
            continue
        h = json.load(path.open())
        ax.plot([r["epoch"] for r in h], [r["mean_total_loss"] for r in h], label=label, linewidth=1.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("mean total loss"); ax.set_yscale("log")
    ax.set_title("FELNet training loss (log scale) — rebalance removes the overlap-term dominance")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig5_felnet_loss.png"); plt.close(fig)
    print(f"  wrote {out / 'fig5_felnet_loss.png'}")


def fig6_felnet_boxes(out: Path, checkpoint: str) -> None:
    model = load_felnet(checkpoint, device="cpu")
    scale = model.config.overlap_scale
    seq = _seq("visdrone", "val", "uav0000077_00720_v")
    info, records = read_sequence(seq)
    record = _busiest_frame(records)
    img = Image.open(resolve_image_path(seq, record.image_path)).convert("RGB")
    anchors = anchor_boxes(64, 2)
    rng = random.Random(0)

    cell, n = 150, 6
    canvas = Image.new("RGB", (cell * n, cell), (24, 26, 30))
    placed = 0
    for obj in record.objects:
        if placed >= n:
            break
        ssi, _win, targets = crop_ssi_with_targets(
            img, obj, record.objects, config=CropConfig(random_shift_px=16.0),
            rng=rng, require_positive_anchor=True,
        )
        pos = targets["confidences"] > 0
        idx = np.nonzero(pos & (targets["track_ids"] == obj.track_id))[0]
        if idx.size == 0:
            continue
        a = int(idx[0])
        x = torch.from_numpy(np.asarray(ssi, np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
        with torch.no_grad():
            o = model(x)
        gt_box = overlap_to_bbox(anchors[a], targets["overlaps"][a].tolist())
        pred_box = overlap_to_bbox(anchors[a], (o["overlap"][0, a].numpy() * scale).tolist())

        tile = ssi.resize((cell, cell), Image.NEAREST)
        d = ImageDraw.Draw(tile)
        k = cell / 64.0
        for box, color in ((gt_box, (60, 220, 120)), (pred_box, (245, 90, 90))):
            bx, by, bw, bh = box
            if bw > 0 and bh > 0:
                d.rectangle([bx * k, by * k, (bx + bw) * k, (by + bh) * k], outline=color, width=3)
        canvas.paste(tile, (placed * cell, 0))
        placed += 1
    _save(canvas, out / "fig6_felnet_boxes.jpg", max_width=1100)


def fig7_embedding_sep(out: Path) -> None:
    # Similarity distributions measured by scripts/eval_felnet.py on held-out val.
    stats = {"original FELNet": (0.758, 0.105, 0.653), "rebalanced FELNet": (0.898, 0.116, 0.782)}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    xs = np.arange(len(stats)); width = 0.34
    same = [v[0] for v in stats.values()]; diff = [v[1] for v in stats.values()]
    ax.bar(xs - width / 2, same, width, label="same identity", color="#3cbe78")
    ax.bar(xs + width / 2, diff, width, label="different identity", color="#e8505b")
    for i, (s, d) in enumerate(zip(same, diff)):
        ax.annotate(f"separation\n{s - d:.3f}", (i, max(s, d) + 0.06), ha="center", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels(stats.keys())
    ax.set_ylabel("mean cosine similarity"); ax.set_ylim(0, 1.15)
    ax.set_title("FELNet embedding discriminability on held-out val")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig7_embedding_sep.png"); plt.close(fig)
    print(f"  wrote {out / 'fig7_embedding_sep.png'}")


def fig8_tracking(out: Path, yolo: str, felnet: str, tau: int = 3) -> None:
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YoloDetector(YOLO(yolo), conf=0.55, iou=0.2, device=device)
    localizer = FELNetLocalizer(load_felnet(felnet, device=device), device=device)
    jmdst = JMDSTTracker(detector, localizer, tau=tau)

    # A denser sequence, sampled late enough that tracklets are confirmed.
    seq = _seq("uavdt", "val", "M0401")
    info, records = read_sequence(seq)
    size = (info.image_width, info.image_height)

    keep_at = [30, 31, 32, 33]  # one detection frame + the tracking frames after it
    panels = []
    for i, record in enumerate(records[: max(keep_at) + 1]):
        img = Image.open(resolve_image_path(seq, record.image_path)).convert("RGB")
        outs = jmdst.process_frame(img, size)
        if i not in keep_at:
            continue
        d = ImageDraw.Draw(img)
        for o in outs:
            x, y, w, h = o.bbox_xywh
            c = PALETTE[o.track_id % len(PALETTE)]
            d.rectangle([x, y, x + w, y + h], outline=c, width=3)
            d.text((x, max(0, y - 12)), f"ID {o.track_id}", fill=c)
        branch = "DETECTION frame" if i % tau == 0 else "tracking frame"
        d.rectangle([0, 0, 250, 22], fill=(0, 0, 0))
        d.text((6, 6), f"t={i}  {branch}", fill=(255, 255, 255))
        panels.append(img)

    if not panels:
        return
    w, h = panels[0].size
    grid = Image.new("RGB", (w * 2, h * 2), (24, 26, 30))
    for i, p in enumerate(panels[:4]):
        grid.paste(p, ((i % 2) * w, (i // 2) * h))
    _save(grid, out / "fig8_tracking.jpg", max_width=1200)


def fig9_tau_sweep(out: Path) -> None:
    path = Path("outputs/ablations_uavdt/results.json")
    if not path.is_file():
        print("  (skipping fig9: ablation results.json not found)")
        return
    rows = sorted([r for r in json.load(path.open()) if r.get("appearance")], key=lambda r: r["tau"])
    tau = [r["tau"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(tau, [r["MOTA"] for r in rows], "o-", label="MOTA", linewidth=1.8)
    ax.plot(tau, [r["HOTA"] for r in rows], "s-", label="HOTA", linewidth=1.8)
    ax.plot(tau, [r["IDF1"] for r in rows], "^-", label="IDF1", linewidth=1.8)
    ax.set_xlabel("detection interval  tau"); ax.set_ylabel("score")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(tau, [r["FPS"] for r in rows], "d--", color="grey", label="FPS")
    ax2.set_ylabel("FPS (grey, dashed)")
    ax.set_title("Accuracy vs speed trade-off across detection interval (UAVDT val)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig9_tau_sweep.png"); plt.close(fig)
    print(f"  wrote {out / 'fig9_tau_sweep.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/presentation")
    p.add_argument("--yolo-visdrone", default="outputs/yolo_runs/retrain_aug/weights/best.pt")
    p.add_argument("--yolo-uavdt", default="outputs/yolo_runs/uavdt_only/weights/best.pt")
    p.add_argument("--felnet", default="outputs/felnet_runs/rebalanced/best.pt")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {out}")

    fig1_dataset_gt(out)
    fig2_ssi_crops(out)
    fig3_yolo_detections(out, args.yolo_visdrone)
    fig4_yolo_curve(out)
    fig5_felnet_loss(out)
    fig6_felnet_boxes(out, args.felnet)
    fig7_embedding_sep(out)
    fig8_tracking(out, args.yolo_uavdt, args.felnet)
    fig9_tau_sweep(out)
    print("Done.")


if __name__ == "__main__":
    main()
