"""Evaluate a trained FELNet checkpoint on held-out data.

Reports, on the given split (default: val, so this checks generalization
rather than memorization of the training episodes):
  - the same three losses used in training (overlap/confidence/embedding)
  - anchor confidence classification accuracy (paper's >0.9 selection rule)
  - mean IoU between the decoded box (Eq. 2) at the true center anchor and
    the ground-truth SSI box -- a direct localization-quality number
  - mean cosine similarity for same-identity vs different-identity pairs,
    the actual discriminability signal the embedding branch needs to be
    useful for re-identification in Phase 8 (modified DeepSORT)

Usage:
    python scripts/eval_felnet.py --checkpoint outputs/felnet_runs/full_run/best.pt --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.data.crops import anchor_boxes
from jmdst.data.felnet_episode import FELNetEpisodeDataset, collate_felnet_episodes
from jmdst.models import FELNet, FELNetConfig, decode_boxes
from jmdst.training.felnet_loss import FELNetLoss, _sample_pairs


def box_iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between (N,4) xywh boxes, elementwise. Clamps degenerate boxes to 0 area."""

    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2].clamp(min=0), a[:, 1] + a[:, 3].clamp(min=0)
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2].clamp(min=0), b[:, 1] + b[:, 3].clamp(min=0)

    ix1, iy1 = torch.maximum(ax1, bx1), torch.maximum(ay1, by1)
    ix2, iy2 = torch.minimum(ax2, bx2), torch.minimum(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--episodes-per-step", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--n-o", type=int, default=8)
    parser.add_argument("--n-s", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=123)  # different from training seed
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_fields = set(FELNetConfig.__dataclass_fields__)
    model = FELNet(FELNetConfig(**{k: v for k, v in checkpoint["config"].items() if k in config_fields}))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')} ({args.checkpoint})")

    dataset = FELNetEpisodeDataset(
        unified_root=args.unified_root,
        dataset=args.dataset,
        split=args.split,
        k_max=args.k_max,
        n_o=args.n_o,
        n_s=args.n_s,
        length=args.num_steps * args.episodes_per_step,
        horizontal_flip=True,
        seed=args.seed,
    )
    print(f"Evaluating on {len(dataset.sequences)} sequences (dataset={args.dataset or 'all'}, split={args.split}).")

    loader = DataLoader(
        dataset,
        batch_size=args.episodes_per_step,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_felnet_episodes,
    )
    criterion = FELNetLoss()

    totals = {"total": 0.0, "overlap": 0.0, "confidence": 0.0, "embedding": 0.0}
    n_batches = 0
    n_ssis = 0
    correct_confidence_calls = 0
    total_confidence_calls = 0
    iou_sum = 0.0
    iou_count = 0
    pos_sims: list[float] = []
    neg_sims: list[float] = []

    gen = torch.Generator(device=device).manual_seed(args.seed)
    anchors = torch.tensor(anchor_boxes(64, 2), dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in loader:
            for key in ("ssi", "overlaps", "confidences", "center_anchor_index", "identity"):
                batch[key] = batch[key].to(device)

            outputs = model(batch["ssi"])
            losses = criterion(outputs, batch, generator=gen)
            for key in totals:
                totals[key] += float(losses[key].item())
            n_batches += 1
            n_ssis += batch["ssi"].shape[0]

            # Confidence classification: does predicted confidence agree with
            # the >0.9 threshold rule against the ground-truth 0/1 label?
            pred_positive = outputs["confidence"] > args.confidence_threshold
            true_positive = batch["confidences"] > 0.5
            correct_confidence_calls += int((pred_positive == true_positive).sum().item())
            total_confidence_calls += true_positive.numel()

            # Localization: decode the box at the *true* center anchor and
            # compare IoU against the ground-truth box at that anchor.
            rows = torch.arange(batch["ssi"].shape[0], device=device)
            center_idx = batch["center_anchor_index"]
            pred_overlap_center = outputs["overlap"][rows, center_idx]
            gt_overlap_center = batch["overlaps"][rows, center_idx]
            anchor_per_sample = anchors[center_idx]
            pred_box = decode_boxes(pred_overlap_center, anchor_per_sample)
            gt_box = decode_boxes(gt_overlap_center, anchor_per_sample)
            ious = box_iou_xywh(pred_box, gt_box)
            iou_sum += float(ious.sum().item())
            iou_count += ious.numel()

            # Discriminability: same-identity vs different-identity similarity.
            rows_center_emb = outputs["embedding"][rows, center_idx]
            idx_a, idx_b, target = _sample_pairs(batch["identity"], 256, 256, generator=gen)
            if idx_a.numel():
                sims = (rows_center_emb[idx_a] * rows_center_emb[idx_b]).sum(dim=-1)
                pos_sims.extend(sims[target > 0].tolist())
                neg_sims.extend(sims[target < 0].tolist())

    print()
    print(f"Evaluated {n_ssis} SSIs over {n_batches} batches on split='{args.split}'.")
    print(f"  mean total loss:      {totals['total']/n_batches:.4f}")
    print(f"  mean overlap loss:    {totals['overlap']/n_batches:.4f}")
    print(f"  mean confidence loss: {totals['confidence']/n_batches:.4f}")
    print(f"  mean embedding loss:  {totals['embedding']/n_batches:.4f}")
    print()
    print(f"  anchor confidence classification accuracy (>{args.confidence_threshold}): "
          f"{correct_confidence_calls/total_confidence_calls:.4f} "
          f"({correct_confidence_calls}/{total_confidence_calls})")
    print(f"  mean IoU (decoded box @ true anchor vs GT box): {iou_sum/iou_count:.4f}  (n={iou_count})")
    print()
    if pos_sims and neg_sims:
        import statistics as st
        print(f"  same-identity  pair similarity: mean={st.mean(pos_sims):.4f}  (n={len(pos_sims)})")
        print(f"  diff-identity  pair similarity: mean={st.mean(neg_sims):.4f}  (n={len(neg_sims)})")
        print(f"  separation (same - diff):       {st.mean(pos_sims) - st.mean(neg_sims):.4f}")
    else:
        print("  WARNING: no embedding pairs sampled -- cannot report discriminability.")


if __name__ == "__main__":
    main()
