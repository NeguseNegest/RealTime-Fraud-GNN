"""Load the Elliptic Bitcoin graph and build leakage-safe snapshots."""

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset


default_data_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "elliptic"
train_period = (1, 30) #  i choose 1-->30 
validation_period = (31, 34) # small need to mention in limitations
test_period = (35, 49)


def add_temporal_masks(data, time_step):
    """Attach timesteps and chronological masks to the graph."""
    if time_step.ndim != 1 or time_step.numel() != data.num_nodes:
        raise ValueError("time_step must contain one value per graph node")

    time_step = time_step.to(dtype=torch.long)
    known = data.y != 2
    data.time_step = time_step
    data.train_mask = _period_mask(time_step, train_period) & known
    data.val_mask = _period_mask(time_step, validation_period) & known
    data.test_mask = _period_mask(time_step, test_period) & known
    return data


def build_temporal_snapshot(data, history_end, target_period):
    """Build a historical graph that cannot include future transactions."""
    if not hasattr(data, "time_step"):
        raise ValueError("data must have a time_step attribute")

    target_start, target_end = target_period
    if target_start > target_end:
        raise ValueError("target_period start must not exceed its end")
    if target_end > history_end:
        raise ValueError("target_period cannot extend beyond graph history")

    node_ids = (data.time_step <= history_end).nonzero(as_tuple=False).view(-1)
    if node_ids.numel() == 0:
        raise ValueError("history_end produces an empty graph")

    snapshot = data.subgraph(node_ids)
    snapshot.node_id = node_ids
    snapshot.target_mask = _period_mask(snapshot.time_step, target_period) & (snapshot.y != 2)
    if not bool(snapshot.target_mask.any()):
        raise ValueError("target_period contains no labelled nodes")
    return snapshot


def _period_mask(time_step, period):
    start, end = period
    return (time_step >= start) & (time_step <= end)


def load_elliptic_data(root=default_data_root):
    """Load the Elliptic graph and attach the project splits."""
    dataset = EllipticBitcoinDataset(root=str(root))
    data = dataset[0].clone()
    features = pd.read_csv(dataset.raw_paths[0], header=None)
    time_step = torch.tensor(features[1].to_numpy(), dtype=torch.long)
    return add_temporal_masks(data, time_step)
