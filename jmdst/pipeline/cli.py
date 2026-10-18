"""Shared command-line options for building a ``JMDSTTracker``.

``scripts/run_jmdst.py`` (single process) and ``scripts/edge_node.py`` (the
split edge/ground deployment) both build their tracker through here. One
definition means the two cannot drift apart -- which is the precondition for
the edge/ground equivalence check: over a perfect link with lossless frames,
the split system must reproduce ``run_jmdst.py``'s output byte for byte.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from jmdst.tracking import CameraMotionCompensator

from .jmdst import ConfirmedOutput, Detector, JMDSTTracker, Localizer


def add_tracker_arguments(parser: argparse.ArgumentParser) -> None:
    """Detector post-processing and tracker options (paper defaults)."""

    parser.add_argument("--tau", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-cosine-distance", type=float, default=0.3)
    parser.add_argument(
        "--no-appearance",
        action="store_true",
        help="Ablation: disable FELNet-embedding cascade matching (IoU-only association).",
    )
    parser.add_argument(
        "--byte",
        action="store_true",
        help=(
            "Enable ByteTrack's second association stage: keep detections down to "
            "--byte-low-conf and match them IoU-only to tracks the normal stages left "
            "unmatched. Weak boxes can only continue a track, never start one."
        ),
    )
    parser.add_argument(
        "--byte-low-conf",
        type=float,
        default=0.1,
        help="Lower confidence bound for BYTE's weak pool (--conf stays the strong threshold).",
    )
    parser.add_argument(
        "--byte-iou",
        type=float,
        default=0.5,
        help=(
            "Max IoU distance for the BYTE stage; 0.5 means IoU >= 0.5. Stricter than "
            "stage 2's 0.7, since a weak box must agree spatially before it is trusted."
        ),
    )
    parser.add_argument(
        "--cmc",
        action="store_true",
        help=(
            "Enable ECC camera-motion compensation: align each frame to the previous "
            "one and warp all track states into the current coordinate frame. Targets "
            "the UAV-maneuver failure mode the paper documents in Sec. 3.5.4."
        ),
    )
    parser.add_argument("--cmc-scale", type=float, default=0.25, help="Downscale factor for ECC estimation (speed/accuracy trade).")
    parser.add_argument("--cmc-motion", default="euclidean", choices=["euclidean", "affine"], help="ECC motion model.")
    parser.add_argument(
        "--nsa",
        action="store_true",
        help=(
            "Enable StrongSORT's NSA Kalman filter at DETECTION frames: scale measurement "
            "noise by (1 - YOLO confidence) so confident detections pull the state harder. "
            "The paper specifies no Kalman internals, so this is an opt-in improvement."
        ),
    )
    parser.add_argument(
        "--nsa-tracking",
        action="store_true",
        help=(
            "Also apply NSA on tracking-branch updates, using FELNet's anchor confidence. "
            "Measured worse (-1.6 HOTA on both datasets, 2026-08-10): that confidence "
            "reports anchor coverage, not localization accuracy, so it is the wrong signal. "
            "Kept for reproducibility of that ablation."
        ),
    )


def detector_conf(args: argparse.Namespace) -> float:
    """Confidence floor the detector itself must apply.

    With BYTE the detector must emit the weak band too; JMDSTTracker splits
    them at args.conf, so the strong path is bit-identical to BYTE-off.
    """

    return min(args.byte_low_conf, args.conf) if args.byte else args.conf


def tracker_from_args(args: argparse.Namespace, detector: Detector, localizer: Localizer) -> JMDSTTracker:
    """A fresh tracker for one sequence (models are reused across sequences)."""

    return JMDSTTracker(
        detector, localizer, tau=args.tau,
        max_cosine_distance=args.max_cosine_distance,
        use_appearance=not args.no_appearance,
        use_nsa=args.nsa,
        nsa_tracking_branch=args.nsa_tracking,
        use_byte=args.byte,
        byte_conf_high=args.conf,
        byte_iou_distance=args.byte_iou,
        cmc=CameraMotionCompensator(
            scale=args.cmc_scale, motion_type=args.cmc_motion
        ) if args.cmc else None,
    )


def mot_lines(frame_id: int, outputs: Sequence[ConfirmedOutput]) -> list[str]:
    """MOT-Challenge rows: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1."""

    lines = []
    for out in outputs:
        x, y, w, h = out.bbox_xywh
        lines.append(f"{frame_id},{out.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{out.confidence:.4f},-1,-1,-1")
    return lines
