"""Tem2-KAN reference adapter.

The legacy project used the external `kan` package with architectures such as
KAN(width=[300, 32, 64, 32, 20], k=10, grid=10). This module preserves that
published/legacy formulation as a distinct comparator rather than silently
relabeling TrustKAN as Tem2-KAN.
"""
from __future__ import annotations

from torch import nn


class Tem2KANReference(nn.Module):
    """Adapter around the original pykan-style KAN implementation.

    Input is [batch, time, features]. For the original univariate CET setting,
    time and feature dimensions are flattened before the KAN mapping.
    """

    def __init__(self, history: int, n_features: int, horizon: int,
                 hidden=(32, 64, 32), k: int = 10, grid: int = 10):
        super().__init__()
        try:
            from kan import KAN
        except ImportError as exc:
            raise ImportError(
                "Tem2KANReference requires the optional `pykan`/`kan` package. "
                "Install the version documented for the reproduction environment."
            ) from exc
        width = [history * n_features, *hidden, horizon]
        self.model = KAN(width=width, k=k, grid=grid)

    def forward(self, x):
        return self.model(x.flatten(1))
