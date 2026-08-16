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

    REFERENCE_IO = (300, 1, 20)

    def __init__(self, history: int, n_features: int, horizon: int,
                 hidden=(32, 64, 32), k: int = 10, grid: int = 10,
                 *, strict: bool = True, seed: int = 1):
        super().__init__()
        if strict and (history, n_features, horizon) != self.REFERENCE_IO:
            raise ValueError(
                "Strict Tem2-KAN reproduction requires history=300, "
                "n_features=1 and horizon=20"
            )
        try:
            from kan import KAN
        except ImportError as exc:
            raise ImportError(
                "Tem2KANReference requires the optional `pykan`/`kan` package. "
                "Install the version documented for the reproduction environment."
            ) from exc
        width = [history * n_features, *hidden, horizon]
        reference_width = list(width)
        self.model = KAN(width=width, k=k, grid=grid, seed=seed, auto_save=False)
        self.reproduction_spec = {
            "width": reference_width,
            "k": k,
            "grid": grid,
            "seed": seed,
            "strict": strict,
        }

    def forward(self, x):
        return self.model(x.flatten(1))
