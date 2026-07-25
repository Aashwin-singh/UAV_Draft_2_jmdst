from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_data_pipeline import _make_visdrone_source

from jmdst.data.crops import overlap_to_bbox
from jmdst.data.converters import convert_visdrone
from jmdst.data.felnet_episode import (
    FELNetEpisodeDataset,
    _flip_targets_horizontally,
    collate_felnet_episodes,
)
from jmdst.models import FELNet
from jmdst.training.felnet_loss import FELNetLoss, embedding_loss, overlap_loss


class FlipTargetTests(unittest.TestCase):
    def test_flip_is_box_consistent(self) -> None:
        # A box in the top-left anchor (index 0), after horizontal flip, should
        # decode to the mirrored box owned by the top-right anchor (index 1).
        from jmdst.data.crops import anchor_boxes, overlap_vector

        anchors = anchor_boxes(64, 2)
        box = [8.0, 10.0, 20.0, 16.0]  # center x=18 -> left half -> anchor 0

        overlaps = np.zeros((4, 4), dtype=np.float32)
        overlaps[0] = overlap_vector(anchors[0], box)
        targets = {
            "overlaps": overlaps,
            "confidences": np.array([1, 0, 0, 0], dtype=np.float32),
            "class_ids": np.array([0, -1, -1, -1], dtype=np.int64),
            "track_ids": np.array([7, -1, -1, -1], dtype=np.int64),
            "anchors_xywh": np.array(anchors, dtype=np.float32),
            "bboxes_xywh": np.zeros((4, 4), dtype=np.float32),
        }

        flipped = _flip_targets_horizontally(targets)

        # The positive anchor moved from 0 to 1.
        self.assertEqual(flipped["confidences"].tolist(), [0, 1, 0, 0])
        self.assertEqual(int(flipped["track_ids"][1]), 7)

        # Decoding the flipped overlap at anchor 1 gives the mirrored box.
        mirrored = overlap_to_bbox(anchors[1], flipped["overlaps"][1].tolist())
        expected_left = 64.0 - (box[0] + box[2])  # mirror x within 64-wide SSI
        self.assertAlmostEqual(mirrored[0], expected_left, places=4)
        self.assertAlmostEqual(mirrored[2], box[2], places=4)  # width preserved
        self.assertAlmostEqual(mirrored[1], box[1], places=4)  # top preserved
        self.assertAlmostEqual(mirrored[3], box[3], places=4)  # height preserved


class LossTests(unittest.TestCase):
    def test_overlap_loss_zero_when_equal(self) -> None:
        pred = torch.rand(6, 4, 4)
        self.assertAlmostEqual(float(overlap_loss(pred, pred.clone())), 0.0, places=5)

    def test_embedding_loss_rewards_correct_similarity(self) -> None:
        # Two identities, 4 samples. For ~0 loss, same-identity pairs need
        # similarity +1 and different-identity pairs need -1, i.e. the two
        # identities' embeddings must be anti-parallel (a and -a).
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        emb = torch.stack([a, a, b, b])  # identities [0,0,1,1]
        identity = torch.tensor([0, 0, 1, 1])
        gen = torch.Generator().manual_seed(0)
        loss_good = embedding_loss(emb, identity, num_positive=2, num_negative=4, generator=gen)
        self.assertLess(float(loss_good), 1e-3)

        # Swapping so same-identity embeddings are anti-parallel -> high loss.
        emb_bad = torch.stack([a, b, a, b])
        gen2 = torch.Generator().manual_seed(0)
        loss_bad = embedding_loss(emb_bad, identity, num_positive=2, num_negative=4, generator=gen2)
        self.assertGreater(float(loss_bad), float(loss_good))

    def test_full_loss_backprops(self) -> None:
        model = FELNet()
        n = 8
        batch = {
            "ssi": torch.randn(n, 3, 64, 64),
            "overlaps": torch.rand(n, 4, 4) * 10,
            "confidences": torch.randint(0, 2, (n, 4)).float(),
            "center_anchor_index": torch.randint(0, 4, (n,)),
            "identity": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        }
        out = model(batch["ssi"])
        losses = FELNetLoss()(out, batch)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in grads))


class EpisodeDatasetTests(unittest.TestCase):
    def test_episode_structure_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_visdrone_source(root)
            unified = root / "unified"
            convert_visdrone(source, unified, split="train")

            ds = FELNetEpisodeDataset(
                unified,
                dataset="visdrone",
                split="train",
                k_max=1,
                n_o=2,
                n_s=2,
                length=4,
                seed=0,
            )
            episode = ds[0]
            m = episode["ssi"].shape[0]
            self.assertGreater(m, 0)
            self.assertEqual(episode["ssi"].shape[1:], (3, 64, 64))
            self.assertEqual(episode["overlaps"].shape, (m, 4, 4))
            self.assertEqual(episode["confidences"].shape, (m, 4))
            self.assertEqual(episode["center_anchor_index"].shape, (m,))
            self.assertEqual(episode["identity"].shape, (m,))
            # Center anchor index is always a valid positive anchor.
            self.assertTrue(bool((episode["center_anchor_index"] >= 0).all()))
            self.assertTrue(bool((episode["center_anchor_index"] < 4).all()))

            merged = collate_felnet_episodes([ds[0], ds[1]])
            self.assertEqual(
                merged["ssi"].shape[0],
                ds[0]["ssi"].shape[0] + ds[1]["ssi"].shape[0],
            )
            # Identity ids stay unique across the two episodes (no overlap).
            n0 = ds[0]["identity"].max().item() + 1
            self.assertGreaterEqual(int(merged["identity"][ds[0]["ssi"].shape[0]:].min()), n0)


if __name__ == "__main__":
    unittest.main()
