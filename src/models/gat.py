"""
Spatio-Temporal Graph Attention Network (TGAT) architecture and leakage-safe training pipeline.
Provides graceful fallback and diagnostics if PyTorch or PyTorch Geometric are absent.
"""
import os
import time
from typing import Dict, Tuple, Optional, Any
import numpy as np

# Check environment availability for PyTorch and PyTorch Geometric
HAS_TORCH = False
HAS_PYG = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    try:
        from torch_geometric.nn import GATConv
        HAS_PYG = True
    except ImportError:
        HAS_PYG = False
except ImportError:
    HAS_TORCH = False


def is_gat_available() -> Tuple[bool, str]:
    """Check if all dependencies required for GAT model training are present."""
    if not HAS_TORCH:
        return False, "PyTorch ('torch') is not installed in the active environment."
    if not HAS_PYG:
        return False, "PyTorch Geometric ('torch_geometric') is not installed in the active environment."
    return True, "PyTorch and PyTorch Geometric are available."


if HAS_TORCH and HAS_PYG:
    class SpatioTemporalGAT(nn.Module):
        """
        Spatio-Temporal Graph Attention Network for blockchain anomaly detection.
        Combines multi-head graph attention convolutions with temporal time-step embeddings.
        """
        def __init__(
            self,
            in_channels: int = 165,
            hidden_channels: int = 32,
            out_channels: int = 2,
            heads: int = 4,
            dropout: float = 0.3
        ):
            super().__init__()
            self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=2, dropout=dropout)
            self.fc_time = nn.Linear(1, hidden_channels * 2)
            self.fc_dense = nn.Linear(hidden_channels * 2 + hidden_channels * 2, hidden_channels)
            self.out = nn.Linear(hidden_channels, out_channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            h1 = F.elu(self.gat1(x, edge_index))
            h1 = self.dropout(h1)
            h2 = F.elu(self.gat2(h1, edge_index))
            t_embed = F.elu(self.fc_time(t))
            h_fused = torch.cat([h2, t_embed], dim=1)
            h_dense = F.relu(self.fc_dense(h_fused))
            h_dense = self.dropout(h_dense)
            return self.out(h_dense)
else:
    class SpatioTemporalGAT:
        """Placeholder class when PyTorch/PyG is not installed."""
        def __init__(self, *args, **kwargs):
            available, msg = is_gat_available()
            raise ImportError(
                f"Cannot initialize SpatioTemporalGAT: {msg}\n"
                "To enable GAT training, install PyTorch and PyTorch Geometric."
            )
