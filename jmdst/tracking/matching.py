"""Association primitives for modified DeepSORT (paper Sec. 2.4 / A.6).

Pure functions used by the Tracker's two-stage association: appearance-based
cascade matching (with Kalman/Mahalanobis gating) for confirmed tracks, then
IoU assignment for the remainder. Kept free of any Track/Tracker state so they
can be unit-tested directly on synthetic boxes and embeddings.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import xywh_to_xyah

# Cost assigned to gated-out (spatially implausible) track/detection pairs.
INFTY_COST = 1e5
# Chi-square 0.95 quantile, 4 measurement DOF -- the standard DeepSORT gating
# threshold on squared Mahalanobis distance.
CHI2INV95_4DOF = 9.4877


def iou_xywh(bbox: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """IoU of one [l, t, w, h] box against an (N, 4) array of xywh candidates."""

    bbox = np.asarray(bbox, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 4)

    bx1, by1 = bbox[0], bbox[1]
    bx2, by2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
    cx1, cy1 = candidates[:, 0], candidates[:, 1]
    cx2, cy2 = candidates[:, 0] + candidates[:, 2], candidates[:, 1] + candidates[:, 3]

    ix1 = np.maximum(bx1, cx1)
    iy1 = np.maximum(by1, cy1)
    ix2 = np.minimum(bx2, cx2)
    iy2 = np.minimum(by2, cy2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    area_b = bbox[2] * bbox[3]
    area_c = candidates[:, 2] * candidates[:, 3]
    union = area_b + area_c - inter
    return np.where(union > 0, inter / union, 0.0)


def appearance_cost_matrix(
    tracks: list,
    detections: list,
    track_indices: list[int],
    detection_indices: list[int],
) -> np.ndarray:
    """Full (n_tracks x n_detections) cosine-distance cost matrix.

    Each track's representative embedding is its most recently updated feature
    (the paper's no-MSFP setting; with MSFP it would be the predicted feature).
    Embeddings are L2-normalized, so cosine distance = 1 - dot product. Rows or
    detections without an embedding get max cost 1.0 so they fall through to IoU.
    """

    cost = np.ones((len(tracks), len(detections)), dtype=np.float64)
    det_emb = {d: detections[d].embedding for d in detection_indices}
    for ti in track_indices:
        track_emb = tracks[ti].embedding_history[-1] if tracks[ti].embedding_history else None
        if track_emb is None:
            continue
        for di in detection_indices:
            emb = det_emb[di]
            if emb is None:
                continue
            cost[ti, di] = 1.0 - float(np.dot(track_emb, emb))
    return cost


def iou_cost_matrix(
    tracks: list,
    detections: list,
    track_indices: list[int],
    detection_indices: list[int],
) -> np.ndarray:
    """Full (n_tracks x n_detections) IoU-distance (1 - IoU) cost matrix."""

    cost = np.ones((len(tracks), len(detections)), dtype=np.float64)
    if not detection_indices:
        return cost
    det_boxes = np.array([detections[d].bbox_xywh for d in detection_indices], dtype=np.float64)
    for ti in track_indices:
        ious = iou_xywh(tracks[ti].bbox_xywh, det_boxes)
        for k, di in enumerate(detection_indices):
            cost[ti, di] = 1.0 - ious[k]
    return cost


def gate_cost_matrix(
    cost_matrix: np.ndarray,
    kalman_filter,
    tracks: list,
    detections: list,
    track_indices: list[int],
    detection_indices: list[int],
    gated_cost: float = INFTY_COST,
    threshold: float = CHI2INV95_4DOF,
) -> np.ndarray:
    """Set spatially implausible track/detection pairs to gated_cost, in place.

    Uses the Kalman filter's Mahalanobis distance in measurement space: pairs
    whose squared distance exceeds the chi-square gate are unreachable given
    the track's predicted motion, so they cannot be an appearance match.
    """

    if not detection_indices:
        return cost_matrix
    measurements = np.array(
        [xywh_to_xyah(np.asarray(detections[d].bbox_xywh, dtype=np.float64)) for d in detection_indices]
    )
    det_index_arr = np.array(detection_indices)
    for ti in track_indices:
        gating_distance = kalman_filter.gating_distance(tracks[ti].mean, tracks[ti].covariance, measurements)
        cost_matrix[ti, det_index_arr[gating_distance > threshold]] = gated_cost
    return cost_matrix


def min_cost_matching(
    cost_matrix: np.ndarray,
    max_distance: float,
    track_indices: list[int],
    detection_indices: list[int],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment over the given subset of a full cost matrix.

    Pairs whose cost exceeds max_distance are rejected (left unmatched), so a
    forced Hungarian pairing never produces an implausible match.
    """

    if not track_indices or not detection_indices:
        return [], list(track_indices), list(detection_indices)

    sub = cost_matrix[np.ix_(track_indices, detection_indices)].copy()
    # Nudge over-threshold entries just past the gate so the solver avoids them
    # when a better option exists, but we still reject any it is forced to pick.
    sub[sub > max_distance] = max_distance + 1e-5

    rows, cols = linear_sum_assignment(sub)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for r, c in zip(rows, cols):
        if sub[r, c] > max_distance:
            continue
        ti, di = track_indices[r], detection_indices[c]
        matches.append((ti, di))
        matched_tracks.add(ti)
        matched_dets.add(di)

    unmatched_tracks = [t for t in track_indices if t not in matched_tracks]
    unmatched_dets = [d for d in detection_indices if d not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def matching_cascade(
    cost_matrix: np.ndarray,
    max_distance: float,
    tracks: list,
    track_indices: list[int],
    detection_indices: list[int],
    cascade_depth: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """DeepSORT age-prioritized cascade: match recently-seen tracks first.

    At each level the subset of tracks with time_since_update == level+1 is
    matched against the still-unmatched detections, giving priority to tracks
    seen more recently (a longer-missed track is less trustworthy).
    """

    unmatched_dets = list(detection_indices)
    matches: list[tuple[int, int]] = []
    for level in range(cascade_depth):
        if not unmatched_dets:
            break
        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if not track_indices_l:
            continue
        matches_l, _, unmatched_dets = min_cost_matching(
            cost_matrix, max_distance, track_indices_l, unmatched_dets
        )
        matches += matches_l

    matched_tracks = {t for t, _ in matches}
    unmatched_tracks = [t for t in track_indices if t not in matched_tracks]
    return matches, unmatched_tracks, unmatched_dets
