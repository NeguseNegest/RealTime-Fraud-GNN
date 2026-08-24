"""Two-layer GraphSAGE encoder and pre-training classifier."""

from torch import nn
from torch.nn import functional
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """Encode transaction topology as normalized dense embeddings."""

    def __init__(self, in_channels, hidden_channels=128, embedding_dim=64, dropout=0.5):
        super().__init__()
        if min(in_channels, hidden_channels, embedding_dim) <= 0:
            raise ValueError("all channel dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, embedding_dim, aggr="mean")

    def forward(self, x, edge_index):
        """Return embeddings without applying a classification softmax."""
        x = self.conv1(x, edge_index)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return functional.normalize(x, p=2, dim=-1)

    def configuration(self):
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
        }


class GraphSAGEClassifier(nn.Module):
    """Attach a temporary classification head for encoder pre-training."""

    def __init__(self, in_channels, hidden_channels=128, embedding_dim=64, dropout=0.5):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, embedding_dim, dropout)
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, x, edge_index):
        embeddings = self.encoder(x, edge_index)
        return self.classifier(embeddings), embeddings
