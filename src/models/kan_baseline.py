"""A plain KAN-style baseline separated from the proposed temporal architecture."""
from __future__ import annotations

import torch
from torch import nn
from .trustkan import GaussianRBFKANLayer


class StandardKANForecaster(nn.Module):
    """Flattened-lag KAN baseline with no temporal convolution or trust modules.

    This deliberately separates the effect of a KAN nonlinear mapping from the
    proposed temporal representation and downstream reliability mechanisms.
    """
    def __init__(self, history: int, n_features: int, horizon: int, hidden_dim=64, grid_size=8):
        super().__init__()
        d = history * n_features
        self.reduce = nn.Linear(d, hidden_dim)
        self.kan1 = GaussianRBFKANLayer(hidden_dim, hidden_dim, grid_size)
        self.kan2 = GaussianRBFKANLayer(hidden_dim, hidden_dim // 2, grid_size)
        self.head = nn.Linear(hidden_dim // 2, horizon)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x.flatten(1))
        z = self.norm1(z + self.kan1(z))
        z = self.norm2(self.kan2(z))
        return self.head(z)
