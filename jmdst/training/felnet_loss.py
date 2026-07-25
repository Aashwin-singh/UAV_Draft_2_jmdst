"""FELNet training losses (paper Eq. 3, 4, 5, 7).

Three RMSE-style terms, combined as L = lambda1*L_E + lambda2*L_o + lambda3*L_C
(paper Eq. 7, all weights 1.0 by default):

    L_o  overlap RMSE over all anchors of all SSIs (Eq. 3)
    L_C  confidence RMSE over all anchors of all SSIs (Eq. 4)
    L_E  embedding RMSE: fast cross-correlation similarity FCC(E_i, E_j)
         against T(i, j) = +1 (same target) / -1 (different) over sampled
         positive/negative pairs (Eq. 5/6)

The paper writes L_o / L_C with a 1/(4*N_o*N_s) normalizer. That constant is
a per-element mean over anchors and SSIs; we implement the standard RMSE
(sqrt of mean squared error over every scalar element), which differs only by
a fixed scalar and does not change the optimum. With L2-normalized embeddings
(FELNet default), the fast cross-correlation of two feature vectors reduces to
their dot product = cosine similarity, bounded in [-1, 1] to match the +/-1
targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _rmse(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2) + 1e-12)


def overlap_loss(pred_overlap: Tensor, target_overlap: Tensor) -> Tensor:
    """L_o (Eq. 3): RMSE over all anchors x all SSIs x 4 overlap components."""

    return _rmse(pred_overlap, target_overlap)


def confidence_loss(pred_confidence: Tensor, target_confidence: Tensor) -> Tensor:
    """L_C (Eq. 4): RMSE over all anchors x all SSIs."""

    return _rmse(pred_confidence, target_confidence)


def _sample_pairs(
    identity: Tensor,
    num_positive: int,
    num_negative: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample positive/negative index pairs from per-SSI identity ids.

    Returns (idx_a, idx_b, target) where target is +1 for same-identity pairs
    and -1 otherwise. Considers only the upper triangle (i < j) to avoid
    self-pairs and duplicates.
    """

    device = identity.device
    n = identity.shape[0]
    row, col = torch.triu_indices(n, n, offset=1, device=device)
    same = identity[row] == identity[col]

    pos_mask = same.nonzero(as_tuple=True)[0]
    neg_mask = (~same).nonzero(as_tuple=True)[0]

    def _take(pool: Tensor, count: int) -> Tensor:
        if pool.numel() == 0 or count <= 0:
            return pool[:0]
        if pool.numel() <= count:
            return pool
        perm = torch.randperm(pool.numel(), generator=generator, device=device)[:count]
        return pool[perm]

    pos = _take(pos_mask, num_positive)
    neg = _take(neg_mask, num_negative)
    picked = torch.cat([pos, neg], dim=0)

    idx_a = row[picked]
    idx_b = col[picked]
    target = torch.cat(
        [torch.ones(pos.numel(), device=device), -torch.ones(neg.numel(), device=device)]
    )
    return idx_a, idx_b, target


def embedding_loss(
    embeddings: Tensor,
    identity: Tensor,
    num_positive: int = 128,
    num_negative: int = 128,
    generator: torch.Generator | None = None,
) -> Tensor:
    """L_E (Eq. 5): RMSE of FCC similarity vs +/-1 target over sampled pairs.

    Args:
        embeddings: (N, D) per-SSI center-target embeddings (L2-normalized).
        identity: (N,) identity id per SSI; equal ids == same target.
        num_positive/num_negative: pairs to sample (paper's N_P / N_N).

    Returns a scalar loss, or 0 if no pairs could be sampled.
    """

    idx_a, idx_b, target = _sample_pairs(identity, num_positive, num_negative, generator)
    if idx_a.numel() == 0:
        return embeddings.sum() * 0.0

    # FCC = dot product of (already L2-normalized) embeddings == cosine sim.
    similarity = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=-1)
    return _rmse(similarity, target)


@dataclass
class FELNetLoss:
    """Weighted FELNet loss (paper Eq. 7)."""

    lambda_embedding: float = 1.0
    lambda_overlap: float = 1.0
    lambda_confidence: float = 1.0
    num_positive: int = 128
    num_negative: int = 128

    def __call__(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        l_o = overlap_loss(outputs["overlap"], batch["overlaps"])
        l_c = confidence_loss(outputs["confidence"], batch["confidences"])

        n = outputs["embedding"].shape[0]
        rows = torch.arange(n, device=outputs["embedding"].device)
        center_emb = outputs["embedding"][rows, batch["center_anchor_index"]]
        l_e = embedding_loss(
            center_emb,
            batch["identity"],
            self.num_positive,
            self.num_negative,
            generator=generator,
        )

        total = (
            self.lambda_embedding * l_e
            + self.lambda_overlap * l_o
            + self.lambda_confidence * l_c
        )
        return {"total": total, "overlap": l_o, "confidence": l_c, "embedding": l_e}


def felnet_total_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Tensor]:
    """Convenience wrapper: FELNetLoss with the given (E, o, C) weights."""

    loss = FELNetLoss(
        lambda_embedding=weights[0],
        lambda_overlap=weights[1],
        lambda_confidence=weights[2],
    )
    return loss(outputs, batch)
