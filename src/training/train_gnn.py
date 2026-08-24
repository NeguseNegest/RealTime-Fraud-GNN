"""Here I train GraphSAGE on temporal graph snapshots."""

import argparse
from pathlib import Path
import sys

import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional
from torch_geometric.loader import NeighborLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import (
    build_temporal_snapshot,
    default_data_root,
    load_elliptic_data,
    train_period,
    validation_period,
)
from src.model.graphsage import GraphSAGEClassifier, GraphSAGEEncoder


default_checkpoint_path = Path(__file__).resolve().parents[2] / "artifacts" / "graphsage_encoder.pt"


def create_neighbor_loader(snapshot, num_neighbors=(25, 10), batch_size=512, shuffle=False):
    """Sample labelled seed nodes and two hops of historical neighbors."""
    return NeighborLoader(
        snapshot,
        input_nodes=snapshot.target_mask,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        shuffle=shuffle,
        time_attr="time_step",
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_seed_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch.x, batch.edge_index)
        loss = functional.cross_entropy(logits[: batch.batch_size], batch.y[: batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        total_seed_nodes += batch.batch_size
    return total_loss / total_seed_nodes


@torch.no_grad()
def evaluate_pr_auc(model, loader, device):
    """Evaluate illicit-class average precision over sampled seed nodes."""
    model.eval()
    probabilities = []
    labels = []

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index)
        probabilities.append(torch.softmax(logits[: batch.batch_size], dim=1)[:, 1].cpu())
        labels.append(batch.y[: batch.batch_size].cpu())
    return float(average_precision_score(torch.cat(labels).numpy(), torch.cat(probabilities).numpy()))


def train_graphsage(
    data,
    epochs=10,
    batch_size=512,
    num_neighbors=(25, 10),
    hidden_channels=128,
    embedding_dim=64,
    dropout=0.5,
    learning_rate=0.001,
    weight_decay=5e-4,
    seed=42,
    device="auto",
):
    """Train GraphSAGE and restore the model from its best validation epoch."""
    device = resolve_device(device)
    torch.manual_seed(seed)
    train_snapshot = build_temporal_snapshot(data, train_period[1], train_period)
    validation_snapshot = build_temporal_snapshot(data, validation_period[1], validation_period)
    train_loader = create_neighbor_loader(train_snapshot, num_neighbors, batch_size, shuffle=True)
    validation_loader = create_neighbor_loader(validation_snapshot, num_neighbors, batch_size)
    model = GraphSAGEClassifier(data.num_node_features, hidden_channels, embedding_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_epoch = 0
    best_val_pr_auc = float("-inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_pr_auc = evaluate_pr_auc(model, validation_loader, device)
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
        if val_pr_auc > best_val_pr_auc:
            best_epoch = epoch
            best_val_pr_auc = val_pr_auc
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    return {"model": model, "best_epoch": best_epoch, "best_val_pr_auc": best_val_pr_auc}


def save_encoder_checkpoint(encoder, path=default_checkpoint_path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "encoder_state_dict": {name: value.detach().cpu() for name, value in encoder.state_dict().items()},
        "encoder_config": encoder.configuration(),
    }
    torch.save(checkpoint, path)
    return path


def load_encoder_checkpoint(path=default_checkpoint_path, device="cpu"):
    device = resolve_device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    encoder = GraphSAGEEncoder(**checkpoint["encoder_config"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    return encoder.to(device)


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train GraphSAGE on the Elliptic transaction graph.")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", type=int, nargs=2, default=(25, 10), metavar=("HOP_1", "HOP_2"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_elliptic_data(args.data_root)
    result = train_graphsage(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_neighbors=tuple(args.num_neighbors),
        device=args.device,
    )
    checkpoint_path = save_encoder_checkpoint(result["model"].encoder, args.checkpoint)
    print(f"Best validation PR-AUC: {result['best_val_pr_auc']:.4f}")
    print(f"Restored epoch: {result['best_epoch']}")
    print(f"Encoder checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
