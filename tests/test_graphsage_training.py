"""Tests for GraphSAGE architecture, training, and checkpoints."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.data.loader import add_temporal_masks
from src.model.graphsage import GraphSAGEClassifier, GraphSAGEEncoder
from src.training.train_gnn import load_encoder_checkpoint, save_encoder_checkpoint, train_graphsage


class GraphSAGEModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(6, 5)
        self.edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])

    def test_encoder_returns_normalized_embeddings(self):
        encoder = GraphSAGEEncoder(5, hidden_channels=8, embedding_dim=4, dropout=0.0)
        embeddings = encoder(self.x, self.edge_index)
        self.assertEqual(embeddings.shape, (6, 4))
        torch.testing.assert_close(embeddings.norm(p=2, dim=1), torch.ones(6), atol=1e-6, rtol=1e-5)

    def test_classifier_keeps_embeddings_separate_from_logits(self):
        model = GraphSAGEClassifier(5, hidden_channels=8, embedding_dim=4, dropout=0.0)
        logits, embeddings = model(self.x, self.edge_index)
        self.assertEqual(logits.shape, (6, 2))
        self.assertEqual(embeddings.shape, (6, 4))


class GraphSAGETrainingTests(unittest.TestCase):
    def make_graph(self):
        torch.manual_seed(11)
        data = Data(
            x=torch.randn(10, 5),
            y=torch.tensor([0, 1, 0, 1, 2, 0, 1, 0, 1, 2]),
            edge_index=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8]]),
        )
        return add_temporal_masks(data, torch.tensor([1, 2, 3, 4, 5, 31, 32, 33, 34, 34]))

    def test_training_and_checkpoint_round_trip(self):
        data = self.make_graph()
        result = train_graphsage(
            data,
            epochs=2,
            batch_size=4,
            num_neighbors=(2, 1),
            hidden_channels=8,
            embedding_dim=4,
            dropout=0.0,
            learning_rate=0.01,
            device="cpu",
            verbose=False,
        )
        self.assertEqual(len(result["history"]), 2)
        self.assertIn(result["best_epoch"], (1, 2))
        self.assertGreaterEqual(result["best_val_pr_auc"], 0.0)
        self.assertLessEqual(result["best_val_pr_auc"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            save_encoder_checkpoint(result["model"].encoder, path, {"best_epoch": result["best_epoch"]})
            restored, metadata = load_encoder_checkpoint(path)

        result["model"].encoder.eval()
        restored.eval()
        with torch.no_grad():
            expected = result["model"].encoder(data.x, data.edge_index)
            actual = restored(data.x, data.edge_index)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metadata["best_epoch"], result["best_epoch"])


if __name__ == "__main__":
    unittest.main()
