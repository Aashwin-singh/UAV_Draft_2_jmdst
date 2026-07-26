from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jmdst.eval import compute_hota, evaluate_clearmot, load_mot_predictions


def _box(x, y, w=20.0, h=20.0):
    return np.array([x, y, w, h], dtype=np.float64)


class ClearMotTests(unittest.TestCase):
    def test_perfect_tracking(self) -> None:
        gt = {"seq": {f: [(1, _box(10, 10)), (2, _box(100, 100))] for f in range(1, 6)}}
        pred = {"seq": {f: [(1, _box(10, 10)), (2, _box(100, 100))] for f in range(1, 6)}}
        results = evaluate_clearmot(gt, pred)
        overall = results["OVERALL"]
        self.assertAlmostEqual(overall.mota, 1.0)
        self.assertAlmostEqual(overall.idf1, 1.0)
        self.assertAlmostEqual(overall.motp, 1.0)  # paper convention: mean IoU
        self.assertEqual(overall.id_switches, 0)
        self.assertEqual(overall.false_positives, 0)
        self.assertEqual(overall.false_negatives, 0)

    def test_missed_detections_lower_mota(self) -> None:
        gt = {"seq": {f: [(1, _box(10, 10))] for f in range(1, 6)}}
        pred = {"seq": {1: [(1, _box(10, 10))], 2: [(1, _box(10, 10))]}}  # only 2 of 5 frames
        overall = evaluate_clearmot(gt, pred)["OVERALL"]
        self.assertEqual(overall.false_negatives, 3)
        self.assertAlmostEqual(overall.mota, 1 - 3 / 5)

    def test_id_switch_counted(self) -> None:
        # Same GT track, but the prediction's id flips halfway -> 1 ID switch.
        gt = {"seq": {f: [(1, _box(10, 10))] for f in range(1, 5)}}
        pred = {"seq": {1: [(7, _box(10, 10))], 2: [(7, _box(10, 10))], 3: [(8, _box(10, 10))], 4: [(8, _box(10, 10))]}}
        overall = evaluate_clearmot(gt, pred)["OVERALL"]
        self.assertEqual(overall.id_switches, 1)


class HotaTests(unittest.TestCase):
    def test_perfect_tracking_hota_one(self) -> None:
        gt = {"seq": {f: [(1, _box(10, 10)), (2, _box(100, 100))] for f in range(1, 6)}}
        pred = {"seq": {f: [(1, _box(10, 10)), (2, _box(100, 100))] for f in range(1, 6)}}
        result = compute_hota(gt, pred)
        self.assertAlmostEqual(result.hota, 1.0, places=6)
        self.assertAlmostEqual(result.deta, 1.0, places=6)
        self.assertAlmostEqual(result.assa, 1.0, places=6)

    def test_empty_predictions_hota_zero(self) -> None:
        gt = {"seq": {f: [(1, _box(10, 10))] for f in range(1, 6)}}
        pred = {"seq": {}}
        result = compute_hota(gt, pred)
        self.assertAlmostEqual(result.hota, 0.0)
        self.assertAlmostEqual(result.deta, 0.0)

    def test_half_missed_deta_but_perfect_assa(self) -> None:
        # Detect the target in only half the frames, but always with the same
        # (correct) id: association is perfect, detection is not.
        gt = {"seq": {f: [(1, _box(10, 10))] for f in range(1, 5)}}
        pred = {"seq": {1: [(1, _box(10, 10))], 3: [(1, _box(10, 10))]}}  # 2 of 4 frames
        result = compute_hota(gt, pred)
        # DetA = TP/(TP+FN+FP) = 2/(2+2+0) = 0.5 at every alpha (boxes identical).
        self.assertAlmostEqual(result.deta, 0.5, places=6)
        # AssA: single (gt,pred) pair, TPA=2, FNA=0 (gt appears 4x but only... )
        # gt appears 4 frames, pred pair matched 2 -> FNA=2; pred appears 2,
        # matched 2 -> FPA=0. A = 2/(2+2+0)=0.5. AssA=0.5.
        self.assertAlmostEqual(result.assa, 0.5, places=6)
        self.assertAlmostEqual(result.hota, 0.5, places=6)

    def test_id_switch_hurts_assa_not_deta(self) -> None:
        # Every frame detected (perfect detection), but the predicted id flips
        # halfway: DetA stays 1, AssA drops, HOTA between.
        gt = {"seq": {f: [(1, _box(10, 10))] for f in range(1, 5)}}
        pred = {"seq": {1: [(7, _box(10, 10))], 2: [(7, _box(10, 10))], 3: [(8, _box(10, 10))], 4: [(8, _box(10, 10))]}}
        result = compute_hota(gt, pred)
        self.assertAlmostEqual(result.deta, 1.0, places=6)
        # Two pairs, each TPA=2. For pair (1,7): FNA = gt_count(4) - 2 = 2,
        # FPA = pred_count(2) - 2 = 0 -> A = 2/4 = 0.5. Same for (1,8).
        # AssA = (2*0.5 + 2*0.5)/4 = 0.5.
        self.assertAlmostEqual(result.assa, 0.5, places=6)
        self.assertLess(result.hota, 1.0)
        self.assertGreater(result.hota, 0.5)  # sqrt(1.0 * 0.5) ~ 0.707

    def test_localization_threshold_matters(self) -> None:
        # Prediction offset so IoU is moderate; at high alpha it stops matching.
        gt = {"seq": {1: [(1, _box(0, 0, 20, 20))]}}
        pred = {"seq": {1: [(1, _box(10, 0, 20, 20))]}}  # IoU = 100/300 ~ 0.333
        low = compute_hota(gt, pred, alphas=np.array([0.3]))
        high = compute_hota(gt, pred, alphas=np.array([0.5]))
        self.assertGreater(low.hota, 0.0)  # matches at alpha=0.3
        self.assertAlmostEqual(high.hota, 0.0)  # no match at alpha=0.5


class LoadPredictionsTests(unittest.TestCase):
    def test_round_trip_mot_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seqA.txt"
            path.write_text("1,5,10.0,20.0,30.0,40.0,0.9,-1,-1,-1\n2,5,11.0,21.0,30.0,40.0,0.8,-1,-1,-1\n", encoding="utf-8")
            preds = load_mot_predictions(tmp)
            self.assertIn("seqA", preds)
            self.assertEqual(len(preds["seqA"][1]), 1)
            tid, box = preds["seqA"][1][0]
            self.assertEqual(tid, 5)
            np.testing.assert_allclose(box, [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
