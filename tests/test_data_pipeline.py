"""Tests for temporal data preparation and baseline metrics."""

import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.baseline_xgboost import classification_metrics, evaluate_thresholds
from src.data.loader import add_temporal_masks, build_temporal_snapshot


class TemporalDataTests(unittest.TestCase):
    def setUp(self):
        self.data = Data(
            x=torch.arange(18, dtype=torch.float32).view(6, 3),
            y=torch.tensor([0, 2, 1, 0, 1, 2]),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 4]]),
        )
        self.time_step = torch.tensor([1, 30, 31, 34, 35, 49])

    def test_temporal_masks_exclude_unknown_labels(self):
        data = add_temporal_masks(self.data, self.time_step)
        self.assertEqual(data.train_mask.tolist(), [True, False, False, False, False, False])
        self.assertEqual(data.val_mask.tolist(), [False, False, True, True, False, False])
        self.assertEqual(data.test_mask.tolist(), [False, False, False, False, True, False])

    def test_snapshot_excludes_future_nodes_and_edges(self):
        data = add_temporal_masks(self.data, self.time_step)
        snapshot = build_temporal_snapshot(data, 34, (31, 34))
        self.assertEqual(snapshot.num_nodes, 4)
        self.assertEqual(snapshot.node_id.tolist(), [0, 1, 2, 3])
        self.assertEqual(snapshot.target_mask.tolist(), [False, False, True, True])
        self.assertLess(int(snapshot.edge_index.max()), snapshot.num_nodes)
        self.assertEqual(snapshot.num_edges, 3)

    def test_snapshot_rejects_future_target_period(self):
        data = add_temporal_masks(self.data, self.time_step)
        with self.assertRaises(ValueError):
            build_temporal_snapshot(data, 30, (31, 34))


class BaselineMetricTests(unittest.TestCase):
    def test_classification_metrics_match_expected_values(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.7, 0.9])
        metrics = classification_metrics(labels, probabilities, 0.5)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        np.testing.assert_array_equal(metrics["confusion_matrix"], np.array([[1, 1], [0, 2]]))

    def test_classification_metrics_validate_threshold(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.array([0, 1]), np.array([0.1, 0.9]), 1.1)

    def test_validation_threshold_comparison(self):
        class Predictor:
            def predict_proba(self, features):
                probabilities = features[:, 0]
                return np.column_stack((1 - probabilities, probabilities))

        data = Data(
            x=torch.tensor([[0.1], [0.8], [0.7], [0.9]]),
            y=torch.tensor([0, 0, 1, 1]),
            val_mask=torch.ones(4, dtype=torch.bool),
        )
        comparison = evaluate_thresholds(Predictor(), data, thresholds=(0.5,))
        self.assertAlmostEqual(comparison[0.5]["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
