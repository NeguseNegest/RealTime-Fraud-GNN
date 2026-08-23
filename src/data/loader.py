"""Load the Elliptic Bitcoin graph with chronological data splits."""

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "elliptic"


def load_elliptic_data(root=DEFAULT_DATA_ROOT):
    dataset = EllipticBitcoinDataset(root=str(root))
    data = dataset[0].clone()

    features = pd.read_csv(dataset.raw_paths[0], header=None)
    time_step = torch.tensor(features[1].to_numpy(), dtype=torch.long)
    known = data.y != 2

    data.time_step = time_step
    data.train_mask = (time_step >= 1) & (time_step <= 30) & known
    data.val_mask = (time_step >= 31) & (time_step <= 34) & known
    data.test_mask = (time_step >= 35) & (time_step <= 49) & known

    return data
