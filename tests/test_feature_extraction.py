from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_data_pipeline import _make_visdrone_source

from jmdst.data.converters import convert_visdrone
from jmdst.data.features import (
    extract_sequence_features,
    group_by_track,
    load_features,
    save_features,
)
from jmdst.data.io import iter_sequence_dirs
from jmdst.models import FELNet, FELNetConfig


class FeatureExtractionTests(unittest.TestCase):
    def test_extract_group_and_roundtrip(self) -> None:
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_visdrone_source(root)
            unified = root / "unified"
            convert_visdrone(source, unified, split="train")
            sequence_dir = iter_sequence_dirs(unified, dataset="visdrone", split="train")[0]

            model = FELNet(FELNetConfig(embedding_dim=16))
            features = extract_sequence_features(model, sequence_dir, device="cpu")

            # The synthetic source has 2 frames, 2 vehicles each (car+bus),
            # so 4 object embeddings, across 2 track ids.
            self.assertEqual(features["embeddings"].shape, (4, 16))
            self.assertEqual(features["track_ids"].shape, (4,))
            self.assertEqual(features["frame_ids"].shape, (4,))
            self.assertEqual(features["boxes_xywh"].shape, (4, 4))

            grouped = group_by_track(features)
            self.assertEqual(len(grouped), 2)
            for track in grouped.values():
                # Each target appears in both frames, ordered by frame id.
                self.assertEqual(track["embeddings"].shape, (2, 16))
                self.assertTrue(np.all(np.diff(track["frame_ids"]) > 0))

            # Embeddings are L2-normalized (FELNet default).
            norms = np.linalg.norm(features["embeddings"], axis=1)
            np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

            # Save/load round-trips exactly.
            path = root / "feat.npz"
            save_features(path, features)
            loaded = load_features(path)
            for key in features:
                np.testing.assert_array_equal(loaded[key], features[key])


if __name__ == "__main__":
    unittest.main()
