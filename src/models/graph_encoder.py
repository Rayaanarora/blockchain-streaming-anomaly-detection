"""
Graph attention encoder for the Elliptic transaction graph.

This is a *static, per-snapshot* GAT: Elliptic edges never cross time steps
(verified: 0 cross-time-step edges), so message passing only mixes
transactions of the same time step. The model does NOT model temporal
evolution and must not be described as a temporal graph network. Unlike the
legacy ``SpatioTemporalGAT`` it takes no scalar time input, which removes the
linear extrapolation of t/49 from training steps (1-29) to test steps (35-49).
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def make_edge_index(edge_index: np.ndarray, direction: str) -> np.ndarray:
    """Return edges for message passing.

    direction="forward": messages flow source -> target (payment direction).
    direction="bidirectional": both directions (adds the reversed edges).
    """
    if direction == "forward":
        return edge_index
    if direction == "bidirectional":
        return np.concatenate([edge_index, edge_index[::-1]], axis=1)
    raise ValueError(f"unknown edge direction: {direction!r}")


class GraphAttentionEncoder(nn.Module):
    """Two GATConv layers + linear head; ``embed`` returns the penultimate layer."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, heads: int = 4,
                 dropout: float = 0.3, embedding_dim: int = 32):
        super().__init__()
        self.config = dict(in_channels=in_channels, hidden_channels=hidden_channels, heads=heads,
                           dropout=dropout, embedding_dim=embedding_dim)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, embedding_dim, heads=1, dropout=dropout)
        self.out = nn.Linear(embedding_dim, 2)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.gat1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.elu(self.gat2(h, edge_index))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.out(self.embed(x, edge_index))

    @torch.no_grad()
    def predict(self, x: np.ndarray, edge_index: np.ndarray, batch_embed: bool = True):
        """Return (illicit probability, embedding) for every node of the given graph."""
        self.eval()
        xt = torch.as_tensor(x, dtype=torch.float32)
        et = torch.as_tensor(edge_index, dtype=torch.long)
        emb = self.embed(xt, et)
        prob = F.softmax(self.out(emb), dim=1)[:, 1]
        return prob.numpy(), emb.numpy()


def load_encoder(state: dict) -> GraphAttentionEncoder:
    model = GraphAttentionEncoder(**state["config"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model
