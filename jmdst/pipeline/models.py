"""Real YOLO/FELNet-backed detector and localizer for the JMDST pipeline.

These adapt the trained models to the callable interfaces JMDSTTracker expects
(see jmdst.pipeline.jmdst). Kept separate from the orchestration so the branch
logic can be tested with lightweight mocks instead of loading models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from jmdst.data.crops import (
    CropConfig,
    anchor_boxes,
    bbox_image_to_ssi,
    bbox_ssi_to_image,
    crop_ssi,
)
from jmdst.data.datasets import _to_tensor
from jmdst.models import FELNet, FELNetConfig, decode_boxes, select_anchor_output


class YoloDetector:
    """Wrap an ultralytics YOLO model as a detector(image) callable.

    Applies the paper's post-processing (NMS IoU 0.2, confidence 0.55 by
    default, Sec. 3.3) and returns [(bbox_xywh, confidence), ...].
    """

    def __init__(self, model, conf: float = 0.55, iou: float = 0.2, imgsz: int = 640, device=None) -> None:
        self.model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def __call__(self, image) -> list[tuple[np.ndarray, float]]:
        results = self.model.predict(
            image, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False
        )
        result = results[0]
        detections: list[tuple[np.ndarray, float]] = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.float64)
            detections.append((np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64), float(box.conf[0])))
        return detections


class FELNetLocalizer:
    """Wrap a trained FELNet as a localizer(image, boxes) callable.

    For each input box: crop its 64x64 SSI, run FELNet, select the anchor via
    the paper's rule (using the box as the reference), and return the selected
    (embedding, refined box in image coords, anchor confidence).
    """

    def __init__(self, felnet: FELNet, crop_config: CropConfig | None = None, device: str = "cpu", confidence_threshold: float = 0.9) -> None:
        self.felnet = felnet.eval()
        self.crop_config = crop_config or CropConfig(random_shift_px=0.0)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.anchors = torch.tensor(anchor_boxes(64, 2), dtype=torch.float32, device=device)

    def __call__(self, image, boxes_xywh) -> list[tuple[np.ndarray, np.ndarray, float]]:
        # Clamp input box sizes to >= 1px so crop_ssi never sees a degenerate
        # box (defensive: a predicted box should be valid, but never crash a
        # long run on a rare Kalman drift).
        prepared = []
        for b in boxes_xywh:
            b = np.asarray(b, dtype=np.float64).copy()
            b[2] = max(b[2], 1.0)
            b[3] = max(b[3], 1.0)
            prepared.append(b)
        boxes_xywh = prepared
        if not boxes_xywh:
            return []

        ssis, windows = [], []
        for box in boxes_xywh:
            ssi, window = crop_ssi(image, box, config=self.crop_config)
            ssis.append(_to_tensor(ssi))
            windows.append(window)

        x = torch.stack(ssis, dim=0).to(self.device)
        with torch.no_grad():
            out = self.felnet(x)
        # Convert the model's overlap output to SSI pixels so anchor selection
        # and box decoding stay in one coordinate space (see
        # FELNetConfig.overlap_scale; 1.0 for pixel-space checkpoints).
        out["overlap"] = out["overlap"] * self.felnet.config.overlap_scale

        box_ssi = torch.tensor(
            [bbox_image_to_ssi(box, window) for box, window in zip(boxes_xywh, windows)],
            dtype=torch.float32, device=self.device,
        )
        selected = select_anchor_output(out["overlap"], out["confidence"], box_ssi, self.anchors, self.confidence_threshold)
        rows = torch.arange(len(boxes_xywh), device=self.device)
        sel_overlap = out["overlap"][rows, selected]
        sel_emb = out["embedding"][rows, selected].cpu().numpy()
        sel_conf = out["confidence"][rows, selected].cpu().numpy()
        refined_ssi = decode_boxes(sel_overlap, self.anchors[selected]).cpu().numpy()

        results = []
        for i, window in enumerate(windows):
            refined_img = np.asarray(bbox_ssi_to_image(refined_ssi[i], window), dtype=np.float64)
            # FELNet's overlap->box decode (Eq. 2) can be degenerate (non-positive
            # width/height) when the crop is off-target and no anchor clears the
            # confidence threshold. Treat that as a failed localization and fall
            # back to the input box (the Kalman-predicted / detection box).
            if refined_img[2] <= 1.0 or refined_img[3] <= 1.0:
                refined_img = boxes_xywh[i].copy()
            results.append((sel_emb[i].astype(np.float64), refined_img, float(sel_conf[i])))
        return results


def load_felnet(checkpoint_path: str | Path, device: str = "cpu") -> FELNet:
    """Load a trained FELNet checkpoint (as saved by scripts/train_felnet.py)."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fields = set(FELNetConfig.__dataclass_fields__)
    model = FELNet(FELNetConfig(**{k: v for k, v in checkpoint["config"].items() if k in fields}))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def build_jmdst(
    yolo_weights: str | Path,
    felnet_checkpoint: str | Path,
    tau: int = 3,
    device: str | None = None,
    conf: float = 0.55,
    iou: float = 0.2,
    imgsz: int = 640,
    **tracker_kwargs,
):
    """Assemble a JMDSTTracker backed by trained YOLO and FELNet models."""

    from ultralytics import YOLO

    from .jmdst import JMDSTTracker

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = YoloDetector(YOLO(str(yolo_weights)), conf=conf, iou=iou, imgsz=imgsz, device=device)
    localizer = FELNetLocalizer(load_felnet(felnet_checkpoint, device=device), device=device)
    return JMDSTTracker(detector, localizer, tau=tau, **tracker_kwargs)
