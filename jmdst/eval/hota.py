"""HOTA / DetA / AssA (paper Sec. 3.2 Eq. 13-15; Luiten et al. 2021).

HOTA(alpha) = sqrt(DetA(alpha) * AssA(alpha)), integrated (averaged) over a
range of localization thresholds alpha. No library in the environment provides
HOTA (motmetrics does not, TrackEval is not installed), so it is implemented
here and unit-tested against known cases.

Detection matching per frame is a Hungarian assignment that maximizes total
IoU, keeping only pairs with IoU >= alpha. Association accuracy is then
computed from the global co-occurrence of each (gt_id, pred_id) matched pair:

    DetA = |TP| / (|TP| + |FN| + |FP|)
    AssA = (1/|TP|) * sum over TP c of  TPA(c) / (TPA(c) + FNA(c) + FPA(c))
      TPA(c) = # frames the pair (gt(c), pred(c)) is matched together
      FNA(c) = (# frames gt(c) appears) - TPA(c)
      FPA(c) = (# frames pred(c) appears) - TPA(c)

IDs are made globally unique across sequences (prefixed by sequence name) so
identities never collide, then all frames are evaluated as one pool -- the
standard "combined" aggregation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from jmdst.tracking.matching import iou_xywh

from .io import FrameDict

DEFAULT_ALPHAS = np.arange(0.05, 1.0, 0.05)  # 0.05, 0.10, ..., 0.95 (19 thresholds)


@dataclass
class HotaResult:
    hota: float
    deta: float
    assa: float
    per_alpha_hota: dict[float, float]


def _global_frames_pair(
    ground_truth: dict[str, FrameDict],
    predictions: dict[str, FrameDict],
) -> tuple[dict[int, list[tuple[str, np.ndarray]]], dict[int, list[tuple[str, np.ndarray]]]]:
    """Merge GT and predictions into two aligned global frame pools.

    Frame ids are offset per sequence so frames from different sequences never
    mix, and identities are prefixed by sequence name so they stay unique.

    The offset MUST be derived from the union of GT and prediction frames for
    each sequence and applied to both: deriving it independently desynchronizes
    the two streams whenever a sequence's frame range differs between them
    (e.g. a sequence with GT frames but an empty prediction file, which happens
    for VisDrone sequences that contain no vehicles at all).
    """

    gt_merged: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
    pred_merged: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)

    offset = 0
    for seq_name in sorted(set(ground_truth) | set(predictions)):
        gt_frames = ground_truth.get(seq_name, {})
        pred_frames = predictions.get(seq_name, {})
        max_frame = max([*gt_frames.keys(), *pred_frames.keys()], default=0)

        for frame_id, objs in gt_frames.items():
            for track_id, box in objs:
                gt_merged[offset + frame_id].append((f"{seq_name}:{track_id}", box))
        for frame_id, objs in pred_frames.items():
            for track_id, box in objs:
                pred_merged[offset + frame_id].append((f"{seq_name}:{track_id}", box))

        offset += max_frame + 1
    return gt_merged, pred_merged


def _match_frame(gt_boxes: np.ndarray, pred_boxes: np.ndarray, alpha: float) -> list[tuple[int, int]]:
    """Hungarian match maximizing IoU; keep pairs with IoU >= alpha."""

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    iou = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float64)
    for g in range(len(gt_boxes)):
        iou[g] = iou_xywh(gt_boxes[g], pred_boxes)
    rows, cols = linear_sum_assignment(-iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= alpha]


def compute_hota(
    ground_truth: dict[str, FrameDict],
    predictions: dict[str, FrameDict],
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> HotaResult:
    """Compute integrated HOTA/DetA/AssA over the given sequences."""

    gt_global, pred_global = _global_frames_pair(ground_truth, predictions)
    all_frames = sorted(set(gt_global) | set(pred_global))

    per_alpha_hota: dict[float, float] = {}
    deta_sum = assa_sum = hota_sum = 0.0

    for alpha in alphas:
        tp = fn = fp = 0
        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        gt_count: dict[str, int] = defaultdict(int)
        pred_count: dict[str, int] = defaultdict(int)

        for frame_id in all_frames:
            gt_objs = gt_global.get(frame_id, [])
            pred_objs = pred_global.get(frame_id, [])
            gt_ids = [i for i, _ in gt_objs]
            pred_ids = [i for i, _ in pred_objs]
            gt_boxes = np.array([b for _, b in gt_objs], dtype=np.float64).reshape(-1, 4)
            pred_boxes = np.array([b for _, b in pred_objs], dtype=np.float64).reshape(-1, 4)

            for gid in gt_ids:
                gt_count[gid] += 1
            for pid in pred_ids:
                pred_count[pid] += 1

            matches = _match_frame(gt_boxes, pred_boxes, alpha)
            matched_gt = {g for g, _ in matches}
            matched_pred = {p for _, p in matches}
            for g, p in matches:
                pair_count[(gt_ids[g], pred_ids[p])] += 1
            tp += len(matches)
            fn += len(gt_ids) - len(matched_gt)
            fp += len(pred_ids) - len(matched_pred)

        deta = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

        if tp > 0:
            ass_sum = 0.0
            for (gid, pid), tpa in pair_count.items():
                fna = gt_count[gid] - tpa
                fpa = pred_count[pid] - tpa
                a_c = tpa / (tpa + fna + fpa)
                ass_sum += tpa * a_c  # each of the tpa TP occurrences contributes a_c
            assa = ass_sum / tp
        else:
            assa = 0.0

        hota_alpha = float(np.sqrt(deta * assa))
        per_alpha_hota[float(alpha)] = hota_alpha
        deta_sum += deta
        assa_sum += assa
        hota_sum += hota_alpha

    n = len(alphas)
    return HotaResult(
        hota=hota_sum / n,
        deta=deta_sum / n,
        assa=assa_sum / n,
        per_alpha_hota=per_alpha_hota,
    )
