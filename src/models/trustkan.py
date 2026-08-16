"""Core TrustKAN-Climate model components.

This initial implementation provides a dependency-light KAN-style temporal model.
It is intentionally modular so uncertainty, drift and selective-prediction modules
can be evaluated independently in ablations.
"""
from __future__ import annotations

import torch
from torch import nn


class GaussianRBFKANLayer(nn.Module):
    """KAN-style layer using learnable univariate Gaussian RBF expansions."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        grid = torch.linspace(-2.5, 2.5, grid_size)
        self.register_buffer("grid", grid)
        self.log_scale = nn.Parameter(torch.zeros(in_features, grid_size))
        self.coeff = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )
        self.base = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        scale = self.log_scale.exp().clamp_min(1e-4)
        basis = torch.exp(
            -0.5 * ((x.unsqueeze(-1) - self.grid) / scale) ** 2
        )
        nonlinear = torch.einsum("...ig,oig->...o", basis, self.coeff)
        return self.base(x) + nonlinear


class TemporalKANBlock(nn.Module):
    """Local temporal mixing followed by an interpretable KAN mapping."""

    def __init__(self, n_features: int, hidden_dim: int, grid_size: int = 8):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.kan = GaussianRBFKANLayer(hidden_dim, hidden_dim, grid_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        z = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(z + self.kan(z))


class TrustKAN(nn.Module):
    """Temporal KAN forecaster with point and quantile outputs.

    Quantile heads enable subsequent conformal calibration. Reliability and OOD
    scoring are deliberately kept outside this class to avoid conflating model
    capacity with post-hoc trust mechanisms in ablation studies.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int = 1,
        hidden_dim: int = 64,
        grid_size: int = 8,
        quantiles=(0.05, 0.5, 0.95),
    ):
        super().__init__()
        self.horizon = horizon
        self.quantiles = tuple(quantiles)
        self.encoder = TemporalKANBlock(n_features, hidden_dim, grid_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.point_head = nn.Linear(hidden_dim, horizon)
        self.quantile_head = nn.Linear(hidden_dim, horizon * len(self.quantiles))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        pooled = self.pool(z.transpose(1, 2)).squeeze(-1)
        point = self.point_head(pooled)
        q = self.quantile_head(pooled).view(
            x.shape[0], self.horizon, len(self.quantiles)
        )
        # Sorting guarantees non-crossing outputs; later work can compare a
        # differentiable monotonic parameterization as an ablation.
        q, _ = torch.sort(q, dim=-1)
        return {"point": point, "quantiles": q, "embedding": pooled}
