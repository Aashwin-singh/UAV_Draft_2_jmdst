from __future__ import annotations

import unittest

import torch

from jmdst.data.crops import anchor_boxes, overlap_vector
from jmdst.models import (
    FELNet,
    FELNetConfig,
    anchor_reference_overlaps,
    decode_boxes,
    select_anchor_output,
)


class FELNetArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = FELNet().eval()

    def test_output_shapes_match_paper_spec(self) -> None:
        batch = 3
        x = torch.randn(batch, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)

        # 2x2 anchor grid = 4 anchors; o is 4-D, E is 16-D, C is scalar.
        self.assertEqual(out["overlap"].shape, (batch, 4, 4))
        self.assertEqual(out["embedding"].shape, (batch, 4, 16))
        self.assertEqual(out["confidence"].shape, (batch, 4))

    def test_backbone_downsamples_64_to_2x2x128(self) -> None:
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            features = self.model.backbone(x)
        # Paper Table 1's final backbone stage: 2x2 spatial, 128 channels.
        self.assertEqual(features.shape, (1, 128, 2, 2))

    def test_backbone_stage_resolutions_follow_table1(self) -> None:
        # Table 1 resolution ladder: 64 -> 32 -> 16 -> 8 -> 4 -> 2.
        x = torch.randn(1, 3, 64, 64)
        seen = []
        with torch.no_grad():
            for layer in self.model.backbone:
                x = layer(x)
                seen.append(x.shape[-1])
        self.assertEqual(seen[0], 64)  # stem conv keeps 64x64
        # Each distinct resolution in order of first appearance.
        ordered_unique = []
        for size in seen:
            if not ordered_unique or ordered_unique[-1] != size:
                ordered_unique.append(size)
        self.assertEqual(ordered_unique, [64, 32, 16, 8, 4, 2])

    def test_confidence_is_bounded_and_embedding_normalized(self) -> None:
        x = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)

        self.assertTrue(bool((out["confidence"] >= 0).all()))
        self.assertTrue(bool((out["confidence"] <= 1).all()))

        norms = out["embedding"].norm(p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))

    def test_embedding_normalization_can_be_disabled(self) -> None:
        model = FELNet(FELNetConfig(normalize_embedding=False)).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64, 64))
        norms = out["embedding"].norm(p=2, dim=-1)
        self.assertFalse(bool(torch.allclose(norms, torch.ones_like(norms))))

    def test_gradients_flow_to_all_three_heads(self) -> None:
        model = FELNet()
        out = model(torch.randn(2, 3, 64, 64))
        loss = out["overlap"].sum() + out["embedding"].sum() + out["confidence"].sum()
        loss.backward()

        for name, head in (
            ("overlap", model.overlap_head),
            ("embedding", model.embedding_head),
            ("confidence", model.confidence_head),
        ):
            grads = [p.grad for p in head.parameters() if p.grad is not None]
            self.assertTrue(grads, f"{name} head received no gradients")
            self.assertTrue(
                any(bool(g.abs().sum() > 0) for g in grads),
                f"{name} head gradients are all zero",
            )

    def test_custom_embedding_dim(self) -> None:
        model = FELNet(FELNetConfig(embedding_dim=32)).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(out["embedding"].shape, (1, 4, 32))


class DecodeAndSelectionTests(unittest.TestCase):
    def test_decode_boxes_inverts_crops_overlap_vector(self) -> None:
        # Cross-module check: the model's Eq. 2 decode must invert the data
        # pipeline's Eq. 1 overlap computation exactly.
        anchors = anchor_boxes(output_size=64, grid_size=2)
        box = [12.0, 20.0, 25.0, 18.0]

        for anchor in anchors:
            overlap = overlap_vector(anchor, box)
            decoded = decode_boxes(
                torch.tensor(overlap, dtype=torch.float32),
                torch.tensor(anchor, dtype=torch.float32),
            )
            torch.testing.assert_close(decoded, torch.tensor(box, dtype=torch.float32))

    def test_decode_boxes_broadcasts_over_batch_and_anchors(self) -> None:
        anchors = torch.tensor(anchor_boxes(64, 2), dtype=torch.float32)
        overlap = torch.rand(5, 4, 4) * 32
        decoded = decode_boxes(overlap, anchors)
        self.assertEqual(decoded.shape, (5, 4, 4))

    def test_reference_overlaps_match_training_target_formula(self) -> None:
        # anchor_reference_overlaps must compute exactly what make_felnet_targets
        # would have used as ground truth, had this box truly belonged to each
        # anchor -- i.e. overlap_vector(anchor, box) per anchor.
        anchors_list = anchor_boxes(64, 2)
        anchors = torch.tensor(anchors_list, dtype=torch.float32)
        box = torch.tensor([10.0, 10.0, 10.0, 10.0])

        ref = anchor_reference_overlaps(box, anchors)
        self.assertEqual(ref.shape, (4, 4))
        for a in range(4):
            expected = overlap_vector(anchors_list[a], box.tolist())
            torch.testing.assert_close(ref[a], torch.tensor(expected, dtype=torch.float32))

    def test_reference_overlaps_batch_broadcast(self) -> None:
        anchors = torch.tensor(anchor_boxes(64, 2), dtype=torch.float32)
        boxes = torch.rand(5, 4) * 20 + 5
        ref = anchor_reference_overlaps(boxes, anchors)
        self.assertEqual(ref.shape, (5, 4, 4))


class SelectAnchorOutputTests(unittest.TestCase):
    """Tests for the paper's Sec. 2.2 inference-time anchor selection rule.

    Each anchor's own reference overlap (what it *should* predict for a given
    candidate box, per anchor_reference_overlaps) trivially has zero distance
    to itself, so a meaningful "nearest anchor" test must deliberately
    perturb the non-target anchors' predicted overlaps away from their own
    reference -- otherwise every anchor is equidistant (0) from its own
    reference and the test can't distinguish selection logic from a tie.
    """

    def setUp(self) -> None:
        self.anchors = torch.tensor(anchor_boxes(64, 2), dtype=torch.float32)

    def test_picks_nearest_confident_anchor(self) -> None:
        box = torch.tensor([[10.0, 10.0, 10.0, 10.0]])  # sits inside anchor 0
        reference = anchor_reference_overlaps(box, self.anchors)  # (1, 4, 4)

        overlap = reference.clone()
        overlap[0, 1] += 0.01  # anchor 1: near-exact match, but low confidence
        overlap[0, 2] += 20.0  # anchor 2: far off, high confidence
        overlap[0, 3] += 20.0  # anchor 3: far off, high confidence
        # anchor 0 left untouched -> exact match (distance 0), high confidence.

        confidence = torch.tensor([[0.95, 0.10, 0.99, 0.99]])
        selected = select_anchor_output(overlap, confidence, box, self.anchors)
        self.assertEqual(selected.tolist(), [0])

    def test_falls_back_when_none_confident(self) -> None:
        box = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
        reference = anchor_reference_overlaps(box, self.anchors)
        confidence = torch.tensor([[0.1, 0.4, 0.8, 0.2]])  # nothing clears 0.9

        selected = select_anchor_output(overlap=reference, confidence=confidence, predicted_box_xywh=box, anchors_xywh=self.anchors)
        self.assertEqual(selected.tolist(), [2])  # falls back to argmax confidence

    def test_handles_mixed_batch(self) -> None:
        # Sample 0's box sits in anchor 0's quadrant, sample 1's in anchor 3's.
        boxes = torch.tensor([[10.0, 10.0, 10.0, 10.0], [40.0, 40.0, 10.0, 10.0]])
        reference = anchor_reference_overlaps(boxes, self.anchors)  # (2, 4, 4)

        overlap = reference.clone()
        # Sample 0: corrupt anchors 1-3 away from their own reference.
        overlap[0, 1:] += 20.0
        # Sample 1: corrupt anchors 0-2 away from their own reference.
        overlap[1, :3] += 20.0

        confidence = torch.tensor([[0.95, 0.99, 0.99, 0.99], [0.99, 0.99, 0.99, 0.95]])
        selected = select_anchor_output(overlap, confidence, boxes, self.anchors)
        self.assertEqual(selected.tolist(), [0, 3])


if __name__ == "__main__":
    unittest.main()
