from pathlib import Path
import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset


default_data_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "elliptic"
local_feature_count = 93
train_period = (1, 30)  # I use timesteps 1-30 for training
validation_period = (31, 34)  # This small validation window is a limitation
test_period = (35, 49)  # test


def add_temporal_masks(data, time_step):
    time_step = time_step.to(dtype=torch.long)
    known = data.y != 2
    data.time_step = time_step
    data.train_mask = _period_mask(time_step, train_period) & known
    data.val_mask = _period_mask(time_step, validation_period) & known
    data.test_mask = _period_mask(time_step, test_period) & known
    return data


def build_temporal_snapshot(data, history_end, target_period):
    node_ids = (data.time_step <= history_end).nonzero(as_tuple=False).view(-1)
    snapshot = data.subgraph(node_ids)
    snapshot.node_id = node_ids
    snapshot.target_mask = _period_mask(snapshot.time_step, target_period) & (snapshot.y != 2)
    return snapshot


def _period_mask(time_step, period):
    start, end = period
    return (time_step >= start) & (time_step <= end)


def load_elliptic_data(root=default_data_root):
    dataset = EllipticBitcoinDataset(root=str(root))
    data = dataset[0].clone()
    data.x = data.x[:, :local_feature_count]
    features = pd.read_csv(dataset.raw_paths[0], header=None)
    time_step = torch.tensor(features[1].to_numpy(), dtype=torch.long)
    return add_temporal_masks(data, time_step)
